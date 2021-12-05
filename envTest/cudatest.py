import torch

testFlag = torch.cuda.is_available()
deviceName = torch.cuda.get_device_name(0)
print(f'Device Available : {testFlag} \n Device Name : {deviceName}')
