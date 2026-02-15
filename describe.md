核心idea在/aifs4su/guhao/MTP/auto_mtp.md

auto_mtp的代码在/aifs4su/guhao/MTP/AutoDeco_vllm/vllm/model_executor/models/llama_eagle3.py
执行should stop的逻辑在/aifs4su/guhao/MTP/AutoDeco_vllm/vllm/v1/spec_decode/eagle.py

启动测试文件在/aifs4su/guhao/MTP/AutoDeco_vllm/slurm.sh，/aifs4su/guhao/MTP/AutoDeco_vllm/test_eagle3.py
我在slurm里启动脚本的，如果需要测试不要直接在当前terminal进行测试

测出来有should stop后比无should stop还慢
log在/aifs4su/guhao/MTP/logs/generation_log/math500_Qwen3-8B-StopHead_disable_should_stop_False_mtp_size_1_greedy_disable_should_stop_False.txt，/aifs4su/guhao/MTP/logs/generation_log/math500_Qwen3-8B-StopHead_disable_should_stop_True_mtp_size_1_greedy_disable_should_stop_True.txt