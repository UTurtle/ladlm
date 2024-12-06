


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
    db_power: -40               # 데시밸 크기
    distribution: "normal"      # 강도에 대한 normal
    time: 5s                    # 시간축으로 중심
    hz: 5000                    # hz 중심
    time_scale: 2               # 얼마나 좌우로 커질지
    hz_scale: 2                 # 얼마나 상하로 커질지
    time_rotate: 90             # 시간을 축으로 90도로 회전
    hz_rotate: 0                # 회전 안함
    gradient_time: 0.5          # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5           # 강도가 gradient하게 감소. 주파수축으로
```


##### 가로선형 패턴도 다음과 같은 방식으로 가능 (물론 세로 선형 패턴도 마찬가지임)
```yaml
shape: "horizen_stripe" # 가로선형
shape_parameters:
    db_power: -80~0
    distribution: "normal"          # 강도에 대한 normal
    time: start_time~end_time
    hz: min_hz~max_hz
    time_scale: abs(gaussian(0,1))
    hz_scale: abs(gaussian(0,1))
    time_rotate: 0
    hz_rotate: 0
    gradient_time: gaussian(0,1)         
    gradient_hz: gaussian(0,1)     
    is_dot_line: false  # 점선인가? 
```


##### 외부 소음 파일 추가 예시
```yaml
shape: "wav_file"
shape_parameters:
    db_power: -40               # 데시밸 크기
    distribution: "normal"      # 강도에 대한 normal
    time: 5s                    # 시간축으로 중심
    hz: 5000                    # hz 중심
    time_scale: 2               # 얼마나 좌우로 커질지
    hz_scale: 2                 # 얼마나 상하로 커질지
    time_rotate: 90             # 시간을 축으로 90도로 회전
    hz_rotate: 0                # 회전 안함
    gradient_time: 0.5          # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5           # 강도가 gradient하게 감소. 주파수축으로
    audio_path: ./audio.wav     # 노이즈 파일
    audio_name: "clap"          # 박수 소리 
```


#### 특정 함수처럼 패턴을 가지는 노이즈를 추가하는 함수


```yaml
pattern: "random_shape_on_range"        # 지정된 구간에 랜덤하게 shape을 생성
pattern_parameter:
    n: 50                   # 50 개 생성
    start_time: 0           # 시작 되는 구간
    end_time: 6.0           # 끝나는 구간
    max_hz: 800             # 최대 Hz
    min_hz: 0               # 최소 Hz
    distribution: "uniform" # 강도에 대한 uniform
    gradient_time: 0.5      # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5       # 강도가 gradient하게 감소. 주파수축으로
    shape: "horizen_stripe"
    shape_parameters:
        db_power: -80~0
        distribution: "normal"          # 강도에 대한 normal
        time: start_time~end_time
        hz: min_hz~max_hz
        time_scale: abs(gaussian(0,1))
        hz_scale: abs(gaussian(0,1))
        time_rotate: 0
        hz_rotate: 0
        gradient_time: gaussian(0,1)         
        gradient_hz: gaussian(0,1)     
        is_dot_line: false  # 점선인가?   
```



```yaml
pattern: "n linear repeat m time sleep" # 특정한 시간적 주기로 반복되는 패턴
pattern_parameter:
    repeat: 3
    repeat_time: 0.5        # 0.5s 마다 반복
    sleep_time: 5           # 3번 반복하고 나서 5s 마다 한번씩 쉼
    start_time: 2.0         # 시작 되는 구간
    end_time: 6.0           # 끝나는 구간
    max_hz: 800             # 최대 Hz
    min_hz: 0               # 최소 Hz
    distribution: "uniform" # 강도에 대한 uniform
    gradient_time: 0.5      # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5       # 강도가 gradient하게 감소. 주파수축으로
    shape: "spike"
    shape_parameters:
        db_power: -80~0
        distribution: "normal"          # 강도에 대한 normal
        time: start_time~end_time
        hz: min_hz~max_hz
        time_scale: abs(gaussian(0,1))
        hz_scale: abs(gaussian(0,1))
        time_rotate: 0
        hz_rotate: 0
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


### time과 hz를 start와 end로 나누어서 보여주는 방법도 고민중


장점
- 범위가 확연하게 보임
- 특히 범위를 지정해서 해당 범위에서 에너지가 강해지거나 낮아짐을 명확히 포착가능

단점
- GPT가 과연 이걸 알아들을 수 있을까? 정보가 너무 많음
- 단순한 shape들에도 적용하기에는 힘들 수 있음.


```yaml
shape: "circle"
shape_parameters:
    db_power: -40               # 데시밸 크기
    distribution: "normal"      # 강도에 대한 normal
    start_time: 4.5             # 시간축 시작
    end_time: 5.5               # 시간축 끝
    start_hz: 4900              # Hz 시작
    end_hz: 5100                # Hz 끝
    time_scale: 2               # 얼마나 좌우로 커질지
    hz_scale: 2                 # 얼마나 상하로 커질지
    time_rotate: 90             # 시간을 축으로 90도로 회전
    hz_rotate: 0                # 회전 안함
    gradient_time: 0.5          # 강도가 gradient하게 증가, 시간축으로
    gradient_hz: -0.5           # 강도가 gradient하게 감소. 주파수축으로
```


```yaml
shape: "horizen_stripe"  # 가로선형
shape_parameters:
    db_power: -40
    distribution: "normal"          # 강도에 대한 분포 (normal)
    start_time: 0.0                 # 시간축 시작
    end_time: 10.0                  # 시간축 끝
    start_hz: 40                    # 주파수축 시작
    end_hz: 50                      # 주파수축 끝
    time_scale: abs(gaussian(0, 1)) # 시간 스케일 (양수)
    hz_scale: abs(gaussian(0, 1))   # 주파수 스케일 (양수)
    time_rotate: 0                  # 시간축 회전
    hz_rotate: 0                    # 주파수축 회전
    gradient_time: gaussian(0, 1)   # 시간축 그라데이션 강도
    gradient_hz: gaussian(0, 1)     # 주파수축 그라데이션 강도
    is_dot_line: false              # 점선 여부
```