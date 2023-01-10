# , _origin없는게 finetuning 용
_base_ = [
    '../models/swinL_upernet_origin.py', '../datasets/trash_datasets_origin2.py',
    '../default_runtime_origin.py', '../schedules/schedule_1x_origin.py'
]