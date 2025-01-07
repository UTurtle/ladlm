


스펙트로그램에 패턴 노이즈를 추가해야함.
하지만 다양한 방식으로 여러가지 패턴을 사용해서 추가 가능해야함.

지금 만들어야 하는 함수는 총 2가지.

- 단순히 노이즈를 해당 위치에 추가하는 함수
- 특정 함수처럼 패턴을 가지는 노이즈를 추가하는 함수

#### 단순히 노이즈를 해당 위치에 추가하는 함수


##### 예시
```yaml
shape: "circle"
shape_parameters:
    db_power_distribution:
        type: "normal"         # 분포 방식 ("normal", "uniform", "none")
            mean: -40              # 평균값 (db_power)
            stddev: 5              # 표준편차 (type이 "normal"일 때 필요)
            min: -80               # 최소값 (type이 "uniform"일 때 필요)
            max: -10               # 최대값 (type이 "uniform"일 때 필요)
    center: (time, hz)
    scale: (time, hz)
    rotation_pivot: (time, hz)
    rotation: 0                 # 회전 안함
    gradient_time: 0.5          # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5           # 강도가 gradient하게 감소. 주파수축으로
```


##### 가로선형 패턴도 다음과 같은 방식으로 가능 (물론 세로 선형 패턴도 마찬가지임)
```yaml
shape: "horizen_stripe" # 가로선형
shape_parameters:
    db_power_distribution:
        type: "normal"         # 분포 방식 ("normal", "uniform", "none")
        mean: -40              # 평균값 (db_power)
        stddev: 5              # 표준편차 (type이 "normal"일 때 필요)
        min: -80               # 최소값 (type이 "uniform"일 때 필요)
        max: -10               # 최대값 (type이 "uniform"일 때 필요)
    center: (time, hz)         # 중심 위치
    scale: (abs(gaussian(0,1)), abs(gaussian(0,1))) # 스케일
    rotation_pivot: (time, hz) # 회전 중심
    rotation: 0                # 회전 각도
    gradient_time: gaussian(0,1)         
    gradient_hz: gaussian(0,1)     
    is_dot_line: false         # 점선 여부

```


##### 외부 소음 파일 추가 예시
```yaml
shape: "wav_file"
shape_parameters:
    db_power_distribution:
        type: "normal"
        mean: -40
        stddev: 5
        min: -80
        max: -10
    center: (time, hz)         # 중심 위치
    scale: (2, 2)              # 스케일
    rotation_pivot: (time, hz) # 회전 중심
    rotation: (90, 0)          # 회전 각도 (time 축: 90도, hz 축: 0도)
    gradient_time: 0.5         # 강도가 시간축으로 gradient하게 증가
    gradient_hz: -0.5          # 강도가 주파수축으로 gradient하게 감소
    audio_path: ./audio.wav    # 노이즈 파일 경로
    audio_name: "clap"         # 노이즈 이름

```


#### 특정 함수처럼 패턴을 가지는 노이즈를 추가하는 함수


```yaml
pattern: "random_shape_on_range"        # 지정된 구간에 랜덤하게 shape 생성
pattern_parameter:
    n: 50                   # 50 개 생성
    start_time: 0           # 시작되는 시간
    end_time: 6.0           # 끝나는 시간
    max_hz: 800             # 최대 Hz
    min_hz: 0               # 최소 Hz
    position_distribution: "uniform" # 위치 분포 (균일)
    gradient_time: 0.5      # 시간축으로 gradient하게 증가
    gradient_hz: -0.5       # 주파수축으로 gradient하게 감소
    shape: "horizen_stripe"
    shape_parameters:
        db_power_distribution:
            type: "normal"
            mean: -40
            stddev: 5
            min: -80
            max: -10
        center: (time, hz)
        scale: (abs(gaussian(0,1)), abs(gaussian(0,1))) # 스케일
        rotation_pivot: (time, hz)
        rotation: 0
        gradient_time: gaussian(0,1)         
        gradient_hz: gaussian(0,1)     
        is_dot_line: false  # 점선 여부  
```



```yaml
pattern: "n linear repeat m time sleep" # 특정 시간 주기로 반복
pattern_parameter:
    repeat: 3
    repeat_time: 0.5        # 0.5s 마다 반복
    sleep_time: 5           # 3번 반복 후 5s 쉼
    time_start: 2.0         # 시작 구간
    time_end: 6.0           # 끝 구간
    freq_max: 800             # 최대 Hz
    freq_min: 0               # 최소 Hz
    position_distribution: "uniform" # 강도에 대한 uniform
    gradient_time: 0.5      # 시간축으로 gradient하게 증가
    gradient_hz: -0.5       # 주파수축으로 gradient하게 감소
    shape: "spike"
    shape_parameters:
        db_power_distribution:
            type: "normal"
            mean: -40
            stddev: 5
            min: -80
            max: -10
        center: (time, hz)
        scale: (abs(gaussian(0,1)), abs(gaussian(0,1))) # 스케일
        rotation_pivot: (time, hz)
        rotation: 0
        gradient_time: gaussian(0,1)      
        gradient_hz: gaussian(0,1)     

```


---


### 현재 목표
   - [ ] Add shape noise (can select)
     - [ ] a circle noise 
     - [ ] a trapezoid
     - [ ] a spike
     - [ ] a pillar
     - [ ] a rectangle
     - [ ] an ellipse
     - [ ] a hill
     - [ ] a fog
     - [ ] a polygon (can selected vertex)
     - [ ] a wave pattern
     - [ ] get real world noise (add wav or mp3 file to merge) 
   - [ ] Add shape noise (can select)
     - [ ] pattern 
       - [ ] linear (time axes or frequency)
       - [ ] random 
       - [ ] n time linear t time sleep (n번 일정하다가 t초 쉬는)
       - [ ] convex  
       - [ ] function ()
     - [ ] type (it can be has parameter)
       - [ ] a circle noise 
       - [ ] a trapezoid
       - [ ] a spike
       - [ ] a pillar
       - [ ] a rectangle
       - [ ] an ellipse
       - [ ] a hill
       - [ ] a fog
       - [ ] a polygon (can selected vertex)
       - [ ] a wave pattern
       - [ ] get real world noise (add wav or mp3 file to merge)




---

### use pipeline - addShape

```python
import numpy as np

# 설정
sample_rate = 16000
duration = 12
n_samples = sample_rate * duration
np.random.seed(42)
signal = np.random.normal(-80, 1, n_samples)  # 기본 신호 생성

# SpectrogramModifier 초기화
spectro_mod = SpectrogramModifier(
    sample_rate=sample_rate, 
    n_fft=1024, 
    hop_length=512, 
    noise_strength=600, 
    noise_type='perlin', 
    noise_params={'seed': 42, 'scale': 100.0}
)

# NoisePipeline 초기화
pipeline = NoisePipeline(spectro_mod)

# Shape 추가
pipeline.addShape(
    shape_name="circle",
    distribution_name="normal",
    shape_params={
        "center": (512, 200),
        "scale": 10,
        "rotation": 0
    },
    dist_params={
        "mean": 0,
        "stddev": 1
    }
)

pipeline.addShape(
    shape_name="horizen_stripe",
    distribution_name="uniform",
    shape_params={
        "start_time": 100,
        "end_time": 300,
        "min_hz": 100,
        "max_hz": 500
    },
    dist_params={
        "min": -1,
        "max": 1
    }
)

# 노이즈가 추가된 스펙트로그램 생성
result = pipeline.generate(signal)

json_data = '''
{
    "shape_name": "circle",
    "distribution_name": "normal",
    "size": [1025, 375],
    "shape_params": {
        "center": [512, 200],
        "scale": 10,
        "rotation": 0
    },
    "dist_params": {
        "mean": 0,
        "stddev": 1
    }
}
'''

pipeline.addShapeFromJSON(json_data)
```



### addPattern

```python
import numpy as np

# 설정
sample_rate = 16000
duration = 12
n_samples = sample_rate * duration
np.random.seed(42)
signal = np.random.normal(-80, 1, n_samples)

# SpectrogramModifier 초기화
spectro_mod = SpectrogramModifier(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    noise_strength=600,
    noise_type='perlin',
    noise_params={'seed': 42, 'scale': 100.0}
)

# NoisePipeline 초기화
pipeline = NoisePipeline(spectro_mod)

# Pattern 추가
pipeline.addPattern(
    "random_shape_on_range",
    {
        "n": 10,
        "freq_min": 500,
        "freq_max": 1000,
        "strength_dB": 40,
        "position_distribution": "uniform",
        "shape": "horizen_sprite",
        "shape_parameters": {
            "db_power_distribution": {
                "type": "constant",
                "power": 5
            },
            "scale": (1, 1), # 기존 스케일 유지
            "rotation": 0
        }
    }
)

# 결과 생성
result = pipeline.generate(signal)
```

