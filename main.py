from losses import *

if __name__ == "__main__":

    # cross entropy with weights
    logits_all_correct = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [4.0, 8.0, 3.0],[4.0, 8.0, 9.0]]
    logits_all_wrong = [[2.0, 4.0, 1.0], [5.0, 2.0, 1.0], [8.0, 4.0, 3.0],[8.0, 2.0, 2.0]]
    logits_common = [[6.0, 1.0, 1.0], [0.0, 5.0, 1.0], [4.0, 8.0, 3.0],[4.0, 8.0, 2.0]] # 1st and 4th are predicted wrongly
    labels = [0, 1, 1, 2]

    ce = CrossEntropyWeightedLoss(weights=[2,2,2])
    print("CE losses are {:.3f},{:.3f},{:.3f}".format(
        ce(labels,logits_all_correct),
        ce(labels,logits_all_wrong),
        ce(labels,logits_common)
    ))

    ce_weighted = CrossEntropyWeightedLoss(weights=[10,2,2])
    print("weighted CE losses are {:.3f},{:.3f},{:.3f}".format(
        ce_weighted(labels,logits_all_correct),
        ce_weighted(labels,logits_all_wrong),
        ce_weighted(labels,logits_common)
    ))

    # dice loss
    loss_dice=DiceLoss()
    print("dice losses are {:.3f},{:.3f},{:.3f}".format(
        loss_dice(labels,logits_all_correct),
        loss_dice(labels,logits_all_wrong),
        loss_dice(labels,logits_common)
    ))


    # focal loss
    loss_focal=FocalLoss()
    print("focal losses are {:.3f},{:.3f},{:.3f}".format(
        loss_focal(labels,logits_all_correct),
        loss_focal(labels,logits_all_wrong),
        loss_focal(labels,logits_common)
    ))

    # trevsky loss
    loss_tv=TverskyLoss()
    print("Tversky losses are {:.3f},{:.3f},{:.3f}".format(
        loss_tv(labels,logits_all_correct),
        loss_tv(labels,logits_all_wrong),
        loss_tv(labels,logits_common)
    ))

    # lovasz loss
    loss_lovasz=LovaszLoss()
    print("lovasz losses are {:.3f},{:.3f},{:.3f}".format(
        loss_lovasz(labels,logits_all_correct),
        loss_lovasz(labels,logits_all_wrong),
        loss_lovasz(labels,logits_common)
    ))
