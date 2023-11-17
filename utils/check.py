import torch

# check if CUDA devices are available
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
else:
    print("CUDA is not available")

# check if CUDA is installed and working
if torch.cuda.is_available():
    try:
        # move a tensor to a CUDA device
        tensor = torch.zeros(1).cuda()
        print("CUDA is installed and working")
    except RuntimeError:
        print("CUDA is installed but not working properly")
else:
    print("CUDA is not available")

# test if a specific CUDA device is available
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        try:
            torch.randn(1, device=device)
            print(f"Device {i} is available")
        except RuntimeError:
            print(f"Device {i} is not available")
else:
    print("CUDA is not available")

# test if mixed-precision training is supported
if torch.cuda.is_available():
    try:
        model = torch.nn.Linear(1, 1).cuda().half()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(10, 1).cuda().half()
        y = torch.randn(10, 1).cuda().half()
        for i in range(10):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print("Mixed-precision training is supported")
    except RuntimeError:
        print("Mixed-precision training is not supported")
else:
    print("CUDA is not available")

# test if CUDA streams are working
if torch.cuda.is_available():
    with torch.cuda.stream(torch.cuda.Stream()):
        x = torch.randn(10000, 10000, requires_grad=True).cuda()
        y = torch.randn(10000, 10000).cuda()
        z = torch.matmul(x, y)
        z.sum().backward()
        torch.cuda.synchronize()
        print("CUDA streams are working")
else:
    print("CUDA is not available")
