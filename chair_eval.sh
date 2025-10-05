MODEL_NAME="./"

echo "========================="
echo "MODEL: $MODEL_NAME"
echo "========================="

# COCO注释与缓存路径需与仓库实际一致
# coco_annotations/ 存放的是 captions/instances 注释；chair_300.pkl 在仓库根目录
python eval/chair.py \
--coco_path ./coco_annotations \
--cache ./chair_300.pkl \
--cap_file $MODEL_NAME/objhal_bench_answer_greedy.jsonl \
--save_path $MODEL_NAME/eval-chair-300_answer.json \
--caption_key answer
