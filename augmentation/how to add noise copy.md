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