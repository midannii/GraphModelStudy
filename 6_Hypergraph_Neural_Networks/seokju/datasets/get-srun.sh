cd ~

# 전체 노드의 상태
sinfo
sinfolong

# 클러스터의 대기열 (queue)상태 확인.
echo $USER ## 현재 사용자의 id
squeue -u $USER
squeuelong -u $USER

# < GPU 4개 이하 사용하는 경우 >
# 예) gpu 1개를 사용할 수 있는 리소스(node) 요청, 사용 시간은 1시간으로 지정.(시간 옵션 필수임)
srun --gres=gpu:1 --time=3:00:00 --pty bash -i

# < GPU 5개 초과사용하는 경우 >
# 예) gpu 10개를 사용할 수 있는 리소스(node) 요청, 사용 시간은 1시간으로 지정.(시간 옵션 필수임)
# srun -p big -q big --gres=gpu:10 --time=1:00:00 --pty bash -i

# 리소스를 할당 받은 후 할당된 작업의 대기열 (queue)상태 재 확인.
echo $USER ## 현재 사용자의 id
squeue -u $USER
squeuelong -u $USER

# 할당 받은 gpu 번호 확인.
echo $CUDA_VISIBLE_DEVICES

# 노드의 gpu 상태 확인.
gpustat

nvidia-smi