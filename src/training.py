













    







# def warmup(current_step: int):
#     return float(current_step / 1000)


scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=linear_schedule_with_warmup(256, 2560))
# train_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=.5)
# scheduler = optim.lr_scheduler.SequentialLR(opt, [warmup_scheduler, train_scheduler], [1000])

lrs1, lrs2 = [], []
step = 0
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        opt.step()
        scheduler.step()
        

        lr1, lr2 = scheduler.get_last_lr() 
        lrs1.append(lr1)
        lrs2.append(lr2)

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')
sns.lineplot(lrs1)
sns.lineplot(lrs2)
10 * (8192/32)
print(scheduler.state_dict().keys())
scheduler.state_dict()['_schedulers']
step
10 ** 0
return 1 / (10 ** (float(1 - current_step)))
min([10 ** float(1 - step) for step in range(1, 100000)])
test = [warmup(step) for step in range(1, 10)]
sns.lineplot(test)
test
scheduler.state_dict()

