from accelerate import Accelerator
import torch_xla.distributed.xla_multiprocessing as xmp


def main():
    # Accelerator instance.
    accelerator = Accelerator()
    print(accelerator.num_processes)

    # Begin training
    # train(opts, accelerator)

    print("OMFG, training finished!")

    # from accelerate import notebook_launcher, Accelerator

    # notebook_launcher(trainer.train, num_processes=8)

    # xmp.spawn(test,(trainer,))

    # trainer.train()

    # trainer.save()


def _mp_fn(index):
    main()


if __name__ == '__main__':
    # main()
    xmp.spawn(_mp_fn, args=())
