## 1. Requirement

- Python 3.7.3
- jieba 0.42.1
- numpy 1.18.5
- tensorflow 2.2.0



## 2. Code

### 2.1. WordFrequency.py

Calculate word frequency and generate a non-sense text. 

### 2.2. write.py

Using LSTM

```bash
1. train from scratch
python3 write.py --train --raw # or
python3 write.py --train --raw --epochs 10 # training epochs 

2. train from pretrained weights
python3 write.py --train # or
python3 write.py --train --epochs 10

3. using trained model
python3 write.py # or
python3 write.py --len 100 # output text length
```

