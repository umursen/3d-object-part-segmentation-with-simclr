from argparse import ArgumentParser

import pytorch_lightning as pl

from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to train', default='shapenet')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')

    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=4, type=int, help="number of GPUs")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--in_features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--final_lr', type=float, default=0.)

    args = parser.parse_args()

    if args.dataset == 'all':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'coseg':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet_toy_dataset':
        dm = ...
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        dataset=args.dataset,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = ...

    trainer = pl.Trainer(
        gpus=args.gpus,
        num_nodes=1,
        precision=16,
        max_epochs=args.num_epochs,
        distributed_backend='ddp',
        sync_batchnorm=True if args.gpus > 1 else False,
    )

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
