# 머신러닝을 이용한 암 진단 및 주요 인자 발굴 AI프로그램 개발

## 프로젝트 개요
![image](https://user-images.githubusercontent.com/80587844/218678242-6d3e09b6-9767-41a1-9130-c630bf4b3280.png)

* 주제는 "머신러닝을 이용한 암진단 모델 개발"로 대장암, 신장병, 전립선암 등의 질병을 유전체, 대사체 발현량 데이터를 통해 진단할 수 있는 프로그램 개발과, 각각의 질병에 영향을 미치는 주요 인자들을 발굴을 목표로 하였습니다.  
* 데이터는 NCBI microarray 데이터, 서울대학교 임상약리학과 신장병 환자 대사체 데이터를 전처리하여 이용하였습니다.
* LGBM, XGBoost, RandomForest 모델을 사용하였으며, python의 scikit-learn 라이브러리를 이용해 구현하였습니다.
## 파일 설명
* nephropathy.ipynb: 대사체 데이터 전처리 코드와 이를 이용한 nephropathy(신장병) 진단 모델
* colon_cancer.ipynb: 대장조직 유전체 발현 데이터 전처리 코드와 이를 이용한 대장암 진단 모델
* 
## 최종 보고서
 https://drive.google.com/file/d/1LbUWOcafP46-2V3WvxHym8hsFWN33nYk/view?usp=share_link
