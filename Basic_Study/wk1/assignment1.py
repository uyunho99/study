# 주의: 기존 코드를 수정하지 마세요.
# 주석을 통해 코드를 설명하는 것을 권장합니다.
# 세미 콜론(;)을 통한 코드 이어 붙이기는 금지입니다.

# Problem 1
# 내부 코드는 오로지 한 줄이어야 합니다.
# Hint: max 함수에 대해 알아보세요.
def p1(X):
    pass

#Problem 2
# 내부 코드는 최대 세 줄이어야 합니다.
def p2(idx):
    pass

# Problem 3
# 내부 코드는 최대 네 줄이어야 합니다.
# Hint: sample과 data의 크기가 클 수 있기 때문에 효율적인 자료구조를 사용하세요.
def p3(sample, data):
    pass

# Problem 4
# 내부 코드는 최대 여섯 줄이어야 합니다.
def p4(sample, data):
    pass

class Matrix2d:
    def __init__(self, data):
        assert len(data) > 0
        self.data = [list(row) for row in data]
        self.shape = (len(data), len(data[0]))
        
    def add(self, x):
        assert self.shape == x.shape
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                self.data[i][j]+=x.data[i][j]
    
    def where(self, func):
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                if func(self.data[i][j]):
                    yield i, j
            
    def __eq__(self, img):
        return img.data == self.data and img.shape == self.shape
    
    def __repr__(self):
        return str([list(map(lambda x: round(x, 4), row)) for row in self.data])
    
# Problem 5
# p5 함수가 정상 동작 하게끔 아래 클래스를 구현하세요.
class GrayScaleImg(Matrix2d):
    def __init__(self, data):
        assert all(all(0<=x and x<=1 for x in row) for row in data)
        pass
    
    def ImgAdd(self, img):
        # 입력으로 들어오는 img 또한 GrayScaleImg 객체이며 해당 데이터를 현재 데이터에 더하는 메소드
        assert isinstance(img, GrayScaleImg)
        pass
    
    def Transpose(self):
        # 현재 데이터를 transpose 시키는 메소드
        pass
    
    def FixBrightness(self, threshold, ratio):
        # threshold 보다 높은 부분을 ratio 만큼 곱해주는 메소드. ratio는 0과 1사이 값이라는 것이 보장됩니다.
        assert 0 <= ratio and ratio <= 1
        pass
    
    def Pad(self):
        # Zero-padding을 추가하는 메소드
        pass

        
def p5(data, low, img_to_add, threshold, ratio):
    try:
        result = dict()
        img = GrayScaleImg(data)
        result["lower"] = list(img.where(lambda x: x < low))

        to_add = GrayScaleImg(img_to_add)
        img.ImgAdd(to_add)
        result["add"] = Matrix2d(img.data)

        img.Transpose()
        result["transpose"] = Matrix2d(img.data)

        img.FixBrightness(threshold, ratio)
        result["fix"] = Matrix2d(img.data)

        img.Pad()
        result["pad"] = Matrix2d(img.data)
        return result
    except Exception as e:
        print("[Error]", e)
        return None
    
# Problem 6
# n은 매우 큰 수가 될 수 있음을 유의하세요.
def p6(n):
    pass