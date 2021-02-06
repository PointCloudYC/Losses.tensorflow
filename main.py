from losses import *

if __name__ == "__main__":

    # cross entropy with weights
    logits_all_correct = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [4.0, 8.0, 3.0],[4.0, 8.0, 9.0]]
    logits_all_wrong = [[2.0, 4.0, 1.0], [5.0, 2.0, 1.0], [8.0, 4.0, 3.0],[8.0, 2.0, 2.0]]
    logits_common = [[6.0, 1.0, 1.0], [0.0, 5.0, 1.0], [4.0, 8.0, 3.0],[4.0, 8.0, 2.0]] # 1st and 4th are predicted wrongly
    labels = [0, 1, 1, 2]

    loss_names = ['ce','wce','dice','focal','tversky','lovasz']
    for loss_name in loss_names:
        if loss_name == 'ce':
            loss_fn = CrossEntropyWeightedLoss(weights=[2,2,2])
        elif loss_name == 'wce':
            loss_fn = CrossEntropyWeightedLoss(weights=[10,2,2])
        elif loss_name == 'dice':
            loss_fn = DiceLoss()
        elif loss_name == 'focal':
            loss_fn = FocalLoss()
        elif loss_name == 'tversky':
            loss_fn = TverskyLoss()
        elif loss_name == 'lovasz':
            loss_fn = LovaszLoss()
        else:
            print('not recognized, use ce loss as defualt.')
            loss_fn = CrossEntropyWeightedLoss(weights=[2,2,2])

        print("{} losses are {:.3f},{:.3f},{:.3f}".format(
            loss_name,
            loss_fn(y_true=labels,y_pred=logits_all_correct),
            loss_fn(labels,logits_all_wrong),
            loss_fn(labels,logits_common)
        ))
    
    wce_fn = CrossEntropyWeightedLoss(weights=[1,2,3])
    dice_fn = DiceLoss()
    loss_fn=ComboundLoss((wce_fn,dice_fn),factor=0.8)
    print("{} losses are {:.3f},{:.3f},{:.3f}".format(
        'compound of WCE and Dice',
        loss_fn(y_true=labels,y_pred=logits_all_correct),
        loss_fn(labels,logits_all_wrong),
        loss_fn(labels,logits_common)
    ))
