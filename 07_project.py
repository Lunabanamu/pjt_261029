"""실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""
from pathlib import Path
import csv
import json
import torch
from torch import nn

# 파일 위치를 기준으로 읽으므로 다른 작업 폴더에서 실행해도 됩니다.
base = Path(__file__).resolve().parent #현재 Python 파일이 있는 폴더를 찾음
with (base / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f)) #CSV를 행별 딕셔너리로 읽음
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

#torch.tensor(data, dtype=...): 데이터를 텐서로 변환
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# 기본필수는 CPU 고정, full-batch 80회입니다. test는 Tensor로 만들지 않습니다.
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
history = [[0, loss_fn(model(X_train), y_train).item()]]

model.train() #학습 모드 설정
for epoch in range(1, 121): #epoch는 전체 학습 데이터를 사용해 모델을 학습하는 단위, 1~80 생성으로 80회 학습
    optimizer.zero_grad() #PyTorch는 gradient를 기본적으로 누적하므로, 이전 반복에서 계산한 gradient를 초기화
    loss = loss_fn(model(X_train), y_train) #train 데이터를 예측하고 정답과 비교/ 실제 클래스 레이블을 비교해 손실(loss)을 계산

    # 06번에서 배운 역전파와 파라미터 갱신 두 문장을 연결하세요.
    # 이미 위에서 이전 gradient를 비우고 loss를 계산했습니다.
    
    # TODO B08: 위 요구사항을 읽고 이 블록을 완성하세요.
    loss.backward() #역전파. 손실이 각 파라미터에 얼마나 영향을 받는지 기울기를 계산
    optimizer.step() #파라미터 갱신. lr=0.1은 한 번 갱신할 때의 학습률
    
    # 갱신 후 같은 train 자료의 loss를 기록합니다. 마지막 batch loss와 구분합니다.
    with torch.no_grad(): # 갱신된 파라미터로 train loss를 다시 계산해 기록
        history.append([epoch, loss_fn(model(X_train), y_train).item()])

model.eval() #평가 모드 설정
with torch.no_grad():  
    logits = model(X_valid)
    # 각 행에서 가장 큰 logit의 열 번호를 prediction으로 구하세요.
    # argmax의 dim=1은 클래스 방향입니다. 결과 shape는 (4,)입니다.
    
    # TODO B09: 위 요구사항을 읽고 이 블록을 완성하세요.
    prediction = logits.argmax(dim=1) #logits.argmax(dim=1): 각 행에서 예측 클래스 선택
    
    accuracy = (prediction == y_valid).float().mean().item() #예측과 정답을 비교해 정확도를 계산하고 Python 숫자로 변환
with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval"])
    writer.writerows(history)
metrics = {"device": "cpu", "epochs": 80, "train_rows": 12, "valid_rows": 4,
           "test_used": False, "valid_accuracy": accuracy,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_ids": [r["id"] for r in valid_rows], "actual": y_valid.tolist(),
           "predicted": prediction.tolist()}
(base / "final_metrics.json").write_text(json.dumps( #데이터를 읽기 쉬운 json 문자열로 변환
    metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
(base / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")
print("분할: train 12 / valid 4 / test 4(보관)")
print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
print("valid 정확도:", round(accuracy, 4))
print("valid 예측:", prediction.tolist())
print("생성: gpu_check.txt, training_history.csv, final_metrics.json")


"""range(1, 81) 실행결과

분할: train 12 / valid 4 / test 4(보관)
train loss: 0.514554 -> 0.110675
valid 정확도: 1.0
valid 예측: [0, 0, 1, 1]
"""

#80회 학습 후 Train Loss가 감소하여 모델이 학습 데이터의 패턴을 학습했음을 확인함
#Validation 데이터 4건에서도 모든 항목을 정확하게 분류함
#학습 후 Train Loss가 감소해 학습 데이터에 대한 예측 오차 감소

"""range(1, 121) 실행결과
분할: train 12 / valid 4 / test 4(보관)
train loss: 0.514554 -> 0.072642
valid 정확도: 1.0
valid 예측: [0, 0, 1, 1]
"""

#학습 데이터의 손실은 추가로 감소했으나, 검증 정확도의 개선은 관찰되지 않음
#이미 검증 데이터 4건에서 이미 모든 항목을 정확하게 분류했기 때문
#더 많은 데이터와 별도의 Test 평가를 통해 일반화 성능을 확인할 필요가 있음