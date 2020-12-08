import torch

def load_data():
    pass


# 한 번의 상황으로 여러 개의 학습 데이터를 만들 수 있다.?
def make_batch(states):
    """
    :param states:
    torch.tensor(
    [
    [state1], [state2], [state3], ...
    ]
    )
    :return:
    """
    n_states, _ = states.size()
    input_batch = []
    target_batch = []

    for i, state in enumerate(states):
        if i == 0:
            continue
        input_batch.append(states[i-1]) # 이 전의 state 를 input 으로 기록
        target_batch.append(state) # 결과 state

    return input_batch, target_batch