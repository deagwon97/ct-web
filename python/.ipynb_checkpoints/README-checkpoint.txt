inference.py가 들어있는 폴더에서 터미널로 다음과 같이 inference.py를 실행합니다.

python inference.py -input_dir input -output -models_dir models -window_size 2


1. inference.py에서는 input 폴더 속에 들어있는 .dcm파일들을 불러와 3차원 이미지를 생성합니다.
2. 생성된 3차원 이미지를 통해 L3의 좌표를 추론합니다.
3. 원본 Saggital plane의 이미지(sagittal.png)와 
4. 추론한 L3좌표가 표시된 이미지(sagittal.overray.png)가 output폴더에 저장됩니다.

5.추론한 L3좌표를 기준으로 상하 window_size 의 horizonttal plane을 생성합니다.
6. 생성된 window_size * 2개의 horizontal plane의 muscle, visceral, subcutanoues를 예측합니다.
7. window_size * 2 개의 horizontal plane의 예측값으로 부터 영역별 비율을 계산합니다.
8. 계산된 비율을 meta_data.csv에 저장합니다.

9. 샘플로 L3좌표의 horizontal plane를 horizontal.png로 저장합니다.
9. 샘플로 L3좌표의 horizontal plane의 예측값을 horizontal_overray.png로 저장합니다.
