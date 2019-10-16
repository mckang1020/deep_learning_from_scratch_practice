from ch5.layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
grape = 200
grape_num = 4
tax = 1.11

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_grape_layer = MulLayer()
add_apple_orange = AddLayer()
add_apple_orange_grape = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
grape_price = mul_orange_layer.forward(grape, grape_num)
# all_price = add_apple_orange_grape.forward(apple_price, orange_price)  # 이렇게 하려고했는데 인자가 2개 뿐
apple_orange_price = add_apple_orange.forward(apple_price, orange_price)
all_price = add_apple_orange_grape.forward(apple_orange_price, grape_price)
price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)

dapple_orange_price, dgrape_price = add_apple_orange_grape.backward(dall_price)
dapple_price, dorange_price = add_apple_orange.backward(dapple_orange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dgrape, dgrape_num = mul_orange_layer.backward(dgrape_price)

print(price)
print(dapple_num, dapple, dorange, dorange_num, dgrape_num, dgrape, dtax)