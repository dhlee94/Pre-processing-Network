# Pre-processing-Network

##네트워크 구성
 - MyModel.py 폴더 안에Timm Labrirary를 이용하여 네트워크 구성
 - timm.list_models(pretrainined==True)를 이용하여 model list 확인 가능
 - timm.create_model('model_file_name', pretrained=True)
 - 추후 업데이트 예정
  
##학습 방법
 - pythom train.py 
 - Config file을 이용하여 parameter 값 변경 가능
 - Optimizer, loss function, scheduler 등 getattr 함수 사용 (추후 업데이트 예정)

##전처리방법
 - albumentations 함수 용
 - Normalization, to Tensor 필수 조건
 - GaussianNoise, RandomCrop, RandomHorizontalFlip 등 다양한 argumentations 값 사용 가능

##Dataload
 - 데이터 파일 경로
 - model/dataloader.py 파일을 이용하여

##실행환경
 - Pytorch-GPU, Conda, Pandas 사용
 - GTX-3090 3장 사용하여 학습 진행

##다양한 학습 방법(추후 업데이트 예정)
 - K-Fold 방식 사용 
 - n-splits 변수 변경가능 (Config)
 - use_kfold 변수 변경가능 (Config)
 
##전처리 네트워크
 - Upsampling, Downsampling으로 구성
 - Upsampling, Downsampling를 거치고 나온 아웃풋 이미지를 Sequence Vector로 생성
 - 생성된 값을 이용하여 1채널 이미지 생성
 - 생성된 이미지를 원본 이미지와 Concate 후 Conv 통과
 - 통과된 이미지는 3채널의 이미지로 BackBone Network의 입력 이미지로 
