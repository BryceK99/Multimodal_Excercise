from functools import partial

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *

from mllm.train.inference_logp import get_batch_logps
from typing import Dict, List, Tuple

### Trainer for SFT
class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if not self.args.use_lora:
            outputs = self.model(data = inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data = inputs, use_cache=False)

        if labels is not None:
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:

            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


### Trainers for NCA
class PreferenceTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        return SequentialSampler(self.train_dataset)

    def preference_loss(
        self,
        beta,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        ### ===> TODO: 实现偏好对齐训练 Loss 计算
        chosen_rewards = beta*(policy_chosen_logps-reference_chosen_logps)
        rejected_rewards = beta*(policy_rejected_logps-reference_rejected_logps)
        losses = -torch.log(torch.sigmoid(chosen_rewards))-(torch.log(torch.sigmoid(-chosen_rewards))+torch.log(torch.sigmoid(-rejected_rewards)))/2

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
        ### <===


    def get_beta_and_logps(self, data_dict, model, args):
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')

        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')

        # win_attention_mask = data_dict.pop('win_attention_mask')
        # rej_attention_mask = data_dict.pop('rej_attention_mask')

        ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
        ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
        ref_win_logp = data_dict.pop('ref_win_logp')
        ref_rej_logp = data_dict.pop('ref_rej_logp')
        ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
        ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')

        if args.preference_use_average_logp:
            ref_win_logp = ref_win_avg_logp
            ref_rej_logp = ref_rej_avg_logp

        beta = data_dict.pop('beta')
        images = data_dict.pop('images')
        # print(data_dict.keys())
        if 'win_context_ids' in data_dict.keys():
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
        concatenated_images = images

        # print("forward input keys:", data_dict.keys())
        # print("concatenated_images:", len(concatenated_images), len(concatenated_images[0]))

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')

        data = {
            "input_ids": concatenated_input_ids,
            "image_bound": data_dict['image_bound'],
            "pixel_values": concatenated_images,
            "tgt_sizes": data_dict['tgt_sizes'],
            "position_ids": data_dict['concatenated_position_ids'],
            "attention_mask": concatenated_attention_mask
        }

        ### ===> TODO: 计算训练过程中，模型在正、负样本上的 logp
        # 注意：我们在数据处理中获取了正负样本拼接后的输入信息，需要将拼接后的输出结果还原
        bsz = win_input_ids.size(0)

        if not args.use_lora:
            policy_seq_logp = get_batch_logps(
                model=model,
                data=data,
                labels=concatenated_labels,
                average_log_prob=args.preference_use_average_logp,
            )
        else:
            with model._enable_peft_forward_hooks(**data):
                policy_seq_logp = get_batch_logps(
                    model=model.base_model,
                    data=data,
                    labels=concatenated_labels,
                    average_log_prob=args.preference_use_average_logp,
                )

        if policy_seq_logp.dim() > 1:
            policy_seq_logp = policy_seq_logp.squeeze(-1)
        policy_win_logp = policy_seq_logp[:bsz]
        policy_rej_logp = policy_seq_logp[bsz:]
        ### <===

        # print("trainer: ref_win_logp:", ref_win_logp.dtype, ref_win_logp.item())
        return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta


    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs

        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = self.get_beta_and_logps(
            data_dict, model, self.args)


        losses, chosen_rewards, rejected_rewards = self.preference_loss(
            beta, policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        loss = losses.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{t}/chosen'] = gather_and_do_mean(chosen_rewards)
        metrics[f'rewards_{t}/rejected'] = gather_and_do_mean(rejected_rewards)
        metrics[f'logps_{t}/rejected'] = gather_and_do_mean(policy_rej_logp)
        metrics[f'logps_{t}/chosen'] = gather_and_do_mean(policy_win_logp)
        metrics[f'logps_{t}/ref_rejected'] = gather_and_do_mean(ref_rej_logp)
        metrics[f'logps_{t}/ref_chosen'] = gather_and_do_mean(ref_win_logp)
        metrics[f'rewards_{t}/accuracies'] = gather_and_do_mean(
            reward_accuracies)
        metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - \
            metrics[f'rewards_{t}/rejected']

        self.log(metrics)

        return loss