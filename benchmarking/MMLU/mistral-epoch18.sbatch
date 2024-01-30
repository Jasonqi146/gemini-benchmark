#!/usr/bin/bash
# run_code_server
#
#SBATCH -J epoch18-mistral
#SBATCH --gres gpu:A6000:4
#SBATCH -t 1-00:00:00

cd /home/zqi2/gemini-benchmark/benchmarking/MMLU/cache_epoch18

# Starting the controller
python3 -m fastchat.serve.controller &

# Starting the model worker with the specified model path
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path /data/tir/projects/tir6/bisk/ruiyiwan/selftrain/init-selftrain-round-2-2nd-exp/checkpoint_improve-0_epoch-18 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000 &

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path /data/tir/projects/tir6/bisk/ruiyiwan/selftrain/init-selftrain-round-2-2nd-exp/checkpoint_improve-0_epoch-18 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001 &

CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path /data/tir/projects/tir6/bisk/ruiyiwan/selftrain/init-selftrain-round-2-2nd-exp/checkpoint_improve-0_epoch-18 --controller http://localhost:21001 --port 31002 --worker http://localhost:31002 &

CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path /data/tir/projects/tir6/bisk/ruiyiwan/selftrain/init-selftrain-round-2-2nd-exp/checkpoint_improve-0_epoch-18 --controller http://localhost:21001 --port 31003 --worker http://localhost:31003 &

# Starting the OpenAI API server on host 0.0.0.0 and port 8000
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8013 &

sleep 180

cd /home/zqi2/gemini-benchmark/benchmarking/MMLU

echo "Starting the experiment"

python run_mmlu.py --model_name sotopia/checkpoint_improve-0_epoch-18 --num_examples 5 > mmlu-mistral-epoch18.log 2>&1