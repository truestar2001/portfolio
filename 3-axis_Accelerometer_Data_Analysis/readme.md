# 3축 가속도 센서를 이용한 자세교정 서비스 AI 모델링

## 프로젝트 개요
웨어러블 벨트 개발사 WELT와 함께 진행한 프로젝트로, WELT사의 대표 제품인 가속도 센서가 장착된 벨트를 이용하여 자세 교정 서비스를 개발하였습니다.
실시간으로 전송되는 3축 가속도 데이터를 통해 현재 착용자의 무게중심이 8방향 중 어느 방향으로 향해있는지 측정해주며 이를 통해 착용자는 자세를 교정할 수 있습니다.
x축, y축, z축 가속도 총 3개의 feature로 구성된 가속도 데이터셋을 이용하였으며 MLP, KNN, 부스팅 등의 다양한 모델을 테스트해보았습니다. 
최종 모델은 LGBM으로 선택하였으며 95.11%의 정확도로 클래스를 예측하는데 성공하였습니다.

## 파일 설명
KNN_Weltried.ipynb: knn알고리즘을 이용한 AI모델 코드  
MLP_Weltried.ipynb: MLP를 이용한 AI모델 코드  
WELT_LightGBM_ver1.ipynb: LGBM을 이용한 AI모델 코드  
Preprocessing.ipynb: 전처리 코드  
WELT_LGBM_ver1.pkl: LGBM으로 학습한 최종 모델    

## 프로젝트 최종 보고서
https://drive.google.com/file/d/1ttXLGvFBgm3w88ytuUEuphMThSnB28Jh/view?usp=share_link
