import pywt

data = pywt.data.camera()

LL, (HL, LH, HH) = pywt.dwt2(data, 'haar')

print(type(LL[0][0]))
