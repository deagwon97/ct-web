[수정내용]------------------------------------------------
 예측한 결과의 3차원 정보를 predict_3d.png, predict_mask.npy형태로 저장합니다.

참고) slice를 상하 20장으로 했을때 z_length는 약 130mm입니다.

[실행]------------------------------------------------
inference.py가 들어있는 폴더에서 터미널로 다음과 같이 inference.py를 실행합니다.

python inference.py -input_dir input -output_dir output -models_dir models -z_length 130

[상세]------------------------------------------------

1. inference.py에서는 input 폴더 속에 들어있는 .dcm파일들을 불러와 3차원 이미지를 생성합니다.
2. 생성된 3차원 이미지를 통해 L3의 좌표를 추론합니다.
3. 원본 Saggital plane의 이미지(sagittal.png)와 
4. 추론한 L3좌표가 표시된 이미지(sagittal.overray.png)가 output폴더에 저장됩니다.

5.추론한 L3좌표를 기준으로 상하 z_length/2 길이([mm])의 영역에 해당하는 horizonttal planes을 생성합니다.(N개)
6. 생성된 N개의 horizontal plane의 muscle, visceral, subcutanoues를 예측합니다.
7. N 개의 horizontal plane의 예측값으로 부터 영역별 면적을 계산합니다.
8. 계산된 면적들 중 L3좌표에 해당하는 slice를 slice_df.csv에 저장합니다.

9. N 개의 horizontal plane들에 대한 영역별 면적과, dicom에서 읽은 z축의 pixcel unit으로 부터 
영역별 부피를 계산하여 vol_df.csv에 저장합니다.


10. 샘플로 L3좌표의 horizontal plane를 horizontal.png로 저장합니다.
11. 샘플로 L3좌표의 horizontal plane의 예측값을 horizontal_overray.png로 저장합니다.
12. 샘플로 L3좌표의 horizontal plane의 예측값을 부위별로 저장합니다.

13. 추론한 L3좌표를 기준으로 상하 z_length/2 길이([mm]) 구간의 영역별 예측결과를 predict_3d.png, predict_mask.npy형태로 저장합니다.
- predict.npy : (512,512, {dicom slice 수})
     -1 : 예측하지 않은 영역( L3좌표 상하 z_length/2 길이([mm]) 구간 밖)
      0 : 근육
      1 : 내장지방
      2 : 피하지방
      3 : 배경


