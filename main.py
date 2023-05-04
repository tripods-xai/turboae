__author__ = "yihanjiang"
# update 10/18/2019, code to replicate TurboAE paper in NeurIPS 2019.
# Tested on PyTorch 1.0.
# TBD: remove all non-TurboAE related functions.

from pprint import pprint

import os
import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from trainer import train, validate, test

from numpy import arange
from numpy.random import mtrand

# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def import_enc(args):
    # choose encoder

    if args.encoder == "TurboAE_rate3_rnn":
        from encoders import ENC_interRNN as ENC

    elif args.encoder in ["TurboAE_rate3_cnn", "TurboAE_rate3_cnn_dense"]:
        from encoders import ENC_interCNN as ENC

    elif args.encoder == "turboae_2int":
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == "rate3_cnn":
        from encoders import CNN_encoder_rate3 as ENC

    elif args.encoder in ["TurboAE_rate3_cnn2d", "TurboAE_rate3_cnn2d_dense"]:
        from encoders import ENC_interCNN2D as ENC

    elif args.encoder == "TurboAE_rate3_rnn_sys":
        from encoders import ENC_interRNN_sys as ENC

    elif args.encoder == "TurboAE_rate2_rnn":
        from encoders import ENC_turbofy_rate2 as ENC

    elif args.encoder == "TurboAE_rate2_cnn":
        from encoders import ENC_turbofy_rate2_CNN as ENC  # not done yet

    elif args.encoder in ["Turbo_rate3_lte", "Turbo_rate3_757"]:
        from encoders import ENC_TurboCode as ENC  # DeepTurbo, encoder not trainable.

    elif args.encoder == "rate3_cnn2d":
        from encoders import ENC_CNN2D as ENC

    else:
        print("Unknown Encoder, stop")

    return ENC


def import_dec(args):

    if args.decoder == "TurboAE_rate2_rnn":
        from decoders import DEC_LargeRNN_rate2 as DEC

    elif args.decoder == "TurboAE_rate2_cnn":
        from decoders import DEC_LargeCNN_rate2 as DEC  # not done yet

    elif args.decoder in ["TurboAE_rate3_cnn", "TurboAE_rate3_cnn_dense"]:
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == "turboae_2int":
        from decoders import DEC_LargeCNN2Int as DEC

    elif args.encoder == "rate3_cnn":
        from decoders import CNN_decoder_rate3 as DEC

    elif args.decoder in ["TurboAE_rate3_cnn2d", "TurboAE_rate3_cnn2d_dense"]:
        from decoders import DEC_LargeCNN2D as DEC

    elif args.decoder == "TurboAE_rate3_rnn":
        from decoders import DEC_LargeRNN as DEC

    elif args.decoder == "nbcjr_rate3":  # ICLR 2018 paper
        from decoders import NeuralTurbofyDec as DEC

    elif args.decoder == "rate3_cnn2d":
        from decoders import DEC_CNN2D as DEC

    return DEC


if __name__ == "__main__":
    #################################################
    # load args & setup logger
    #################################################
    identity = str(np.random.random())[2:8]
    print("[ID]", identity)

    # put all printed things to log file
    logfile = open("./logs/" + identity + "_log.txt", "a")
    sys.stdout = Logger("./logs/" + identity + "_log.txt", sys.stdout)

    args = get_args()
    print(args)

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cpu")  # torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    ENC = import_enc(args)
    DEC = import_dec(args)

    # setup interleaver.
    if args.is_interleave == 1:  # fixed interleaver.
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        p_array2 = rand_gen.permutation(arange(args.block_len))

    elif args.is_interleave == 0:
        p_array1 = range(args.block_len)  # no interleaver.
        p_array2 = range(args.block_len)  # no interleaver.
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array2 = rand_gen.permutation(arange(args.block_len))

    print("using random interleaver", p_array1, p_array2)

    if args.encoder == "turboae_2int" and args.decoder == "turboae_2int":
        encoder = ENC(args, p_array1, p_array2)
        decoder = DEC(args, p_array1, p_array2)
    else:
        encoder = ENC(args, p_array1)
        decoder = DEC(args, p_array1)

    # choose support channels
    from channel_ae import Channel_AE

    model = Channel_AE(args, encoder, decoder).to(device)

    # model = Channel_ModAE(args, encoder, decoder).to(device)

    # make the model parallel
    if args.is_parallel == 1:
        model.enc.set_parallel()
        model.dec.set_parallel()

    # weight loading
    if args.init_nw_weight == "default":
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight, map_location=device)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        except:
            model.load_state_dict(pretrained_model, strict=False)

        model.args = args

    print(model)

    if args.examine:
        from pprint import pprint

        dec_to_save = model.dec
        state_dict = dec_to_save.state_dict()
        pprint(list(state_dict.keys()))
        input()
        pprint({k: v.shape for k, v in state_dict.items()})
        input()
        pprint(state_dict)

        print("Exiting")
        exit()

    if args.compare_encoders:
        import sys

        # sys.path.append("../interpreting-deep-codes/")
        print("Doing test of comparing encoder")
        # if not args.use_precomputed_norm_stats:
        # raise ValueError("Need to be using --use_precomputed_norm_stats")
        if not args.no_code_norm:
            raise ValueError("no_code_norm")

        from src.encoders import (
            turboae_binary_exact_nobd,
            turboae_cont_exact_nn,
            turboae_cont_exact_nobd,
        )
        from src.utils import DEFAULT_DEVICE_MANAGER

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        # encoder = turboae_exact_nonsys_bd(TurboAEInterleaver(), block_len=100, delay=4)
        # encoder = turboae_exact_nonsys_bd_window5_delay2(TurboAEInterleaver(), block_len=100)
        num_steps = 100
        # encoder = turboae_cont_exact_nn(num_steps=num_steps, interleaver="turboae")
        encoder = turboae_cont_exact_nobd(
            num_steps=num_steps, interleaver="turboae", delay=4, constraint=None
        )

        batch_size = args.batch_size
        print(batch_size)
        for i in range(100):
            test_input = torch.randint(
                0,
                2,
                size=(batch_size, num_steps, 1),
                dtype=torch.float,
                device=DEFAULT_DEVICE_MANAGER.device,
            )

            # model.enc.num_test_block = 10000000
            # model.enc.mean_scalar = torch.tensor([-0.0215])
            # model.enc.std_scalar = torch.tensor([0.5114])
            tae_output = model.enc(test_input)
            my_output = encoder(test_input[..., 0])

            print(tae_output[0, :20, :])
            print(my_output[0, :20, :])
            breakpoint()

            matches = tae_output == my_output
            if torch.all(matches):
                print(f"{i} passed!")
            else:
                print(f"{(1 - torch.mean(matches.float())) * 100}% mismatches")

        print("Exiting")
        exit()

    if args.compute_bd_functions:
        print("Computing Bd Functions")
        if not args.use_precomputed_norm_stats:
            raise ValueError("Need to be using --use_precomputed_norm_stats")

        import numpy as np
        import tensorflow as tf
        from src.utils import enumerate_binary_inputs

        class Identity(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")

        test_input = enumerate_binary_inputs(9)[..., None].numpy()
        torch_test_input = torch.from_numpy(test_input)
        model.enc.interleaver = Identity()
        model.enc.num_test_block = 10000000
        model.enc.mean_scalar = torch.tensor([-0.0215]).cpu()
        model.enc.std_scalar = torch.tensor([0.5114]).cpu()
        torch_output = (model.enc(torch_test_input) + 1) / 2

        np_output = torch_output.cpu().detach().numpy().astype(np.uint8)
        fp = "./tmp/test_tae.npy"
        print(f"Saving to {fp}")
        np.save(fp, np_output)

        print("Exiting")
        exit()

    if args.save_decoder:
        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        dec_to_save = model.dec
        state_dict = dec_to_save.state_dict()
        state_dict_path = (
            "./tmp/decoders/" + "_".join([identity, model_filename, "dec"]) + ".pt"
        )
        torch.save(dec_to_save.state_dict(), state_dict_path)

        print("Exiting")
        exit()

    if args.test_compare_decoder_torch:
        from pathlib import Path
        from src.constants import TURBOAE_DECODER_PATH
        from src.decoders import TurboAEDecoder
        from src.interleavers import TurboAEInterleaver
        from src.modulation import Normalization
        from src.channels import VariableAWGN

        print("Doing comparision of decoders with pytorch implementation.")

        decoder_path = Path("../interpreting-deep-codes/src") / TURBOAE_DECODER_PATH
        print(f"Loading decoder from {decoder_path}")
        state_dict = torch.load(decoder_path)
        decoder = TurboAEDecoder(
            num_iteration=6,
            num_iter_ft=5,
            dec_num_layer=5,
            dec_num_unit=100,
            dec_kernel_size=5,
            front_pad=False,
            interleaver=TurboAEInterleaver(),
        )
        decoder.pre_initialize(state_dict)
        channel = VariableAWGN(
            snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high
        )

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")

        batch_size = args.batch_size
        for i in range(10):
            encoder_output = model.enc(
                torch.randint(
                    low=0,
                    high=2,
                    size=(batch_size, decoder.source_data_len, 1),
                    dtype=torch.float32,
                )
            )
            test_dec_input = channel(encoder_output)

            original_output = model.dec(test_dec_input)[..., 0]
            new_output = torch.sigmoid(decoder(test_dec_input))

            print((original_output > 0.5).long())
            print((new_output > 0.5).long())

            torch.testing.assert_allclose(new_output, original_output)
            print(f"{i} passed!")

        print("Exiting")
        exit()

    if args.test_compare_encoder_torch:
        from pathlib import Path
        from src.constants import TURBOAE_ENCODER_CONT_PATH
        from src.encoders import ENC_interCNN
        from src.interleavers import TurboAEInterleaver
        from src.modulation import Normalization
        from src.channels import VariableAWGN

        print("Doing comparision of decoders with pytorch implementation.")

        encoder_path = (
            Path("../interpreting-deep-codes/src") / TURBOAE_ENCODER_CONT_PATH
        )
        print(f"Loading encoder from {encoder_path}")
        state_dict = torch.load(encoder_path)
        encoder = ENC_interCNN(
            enc_num_unit=100,
            enc_num_layer=2,
            enc_kernel_size=5,
            interleaver=TurboAEInterleaver(),
            first_pad=False,
            front_pad=False,
        )
        encoder.pre_initialize(state_dict)
        assert encoder.mean == 0
        assert encoder.std == 1
        modulator = Normalization()

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")

        batch_size = args.batch_size
        for i in range(10):
            test_enc_input = torch.randint(
                low=0,
                high=2,
                size=(batch_size, args.block_len),
                dtype=torch.float32,
            )
            original_output = model.enc(test_enc_input[..., None])
            new_output = modulator(encoder(test_enc_input))

            print(original_output)
            print(torch.mean(original_output))
            print(torch.std(original_output))
            print(new_output)
            print(torch.mean(new_output))
            print(torch.std(new_output))

            torch.testing.assert_allclose(new_output, original_output)
            print(f"{i} passed!")

        print("Exiting")
        exit()

    if args.test_channel_implementation:
        from tqdm import trange
        from pathlib import Path
        from src.channels import VariableAWGN
        from src.utils import snr2sigma, snr2sigma_torch, sigma2snr, sigma2snr_torch
        from channels import generate_noise
        from utils import snr_db2sigma, snr_sigma2db
        import numpy as np

        print("Doing comparision of decoders with pytorch implementation.")

        sigma_low = snr2sigma(args.train_dec_channel_low)
        sigma_high = snr2sigma(args.train_dec_channel_high)
        np.testing.assert_almost_equal(
            sigma_low, snr_db2sigma(args.train_dec_channel_low), verbose=True
        )
        np.testing.assert_almost_equal(
            sigma_high, snr_db2sigma(args.train_dec_channel_high), verbose=True
        )
        np.testing.assert_almost_equal(
            sigma2snr(sigma_low), snr_sigma2db(sigma_low), verbose=True
        )
        np.testing.assert_almost_equal(
            sigma2snr(sigma_high), snr_sigma2db(sigma_high), verbose=True
        )

        batch_sigma = (sigma_low - sigma_high) * torch.rand(
            args.batch_size,
        ) + sigma_high
        batch_snr = sigma2snr_torch(batch_sigma)
        torch.testing.assert_allclose(batch_snr, snr_sigma2db(batch_sigma))
        torch.testing.assert_allclose(
            snr2sigma_torch(batch_snr), snr_db2sigma(batch_snr)
        )

        channel = VariableAWGN(
            snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high
        )
        pprint(channel.settings())

        batch_size = args.batch_size
        test_enc_input = torch.randint(
            low=0,
            high=2,
            size=(batch_size, args.block_len, 1),
            dtype=torch.float32,
        )

        bins = torch.linspace(start=sigma_high, end=sigma_low, steps=50)
        enc_output = model.enc(test_enc_input)
        running_mean_original = 0
        running_var_original = 0
        original_count = 0
        original_sigma_hist = 0
        for i in trange(1000):
            original_noise = generate_noise(
                (args.batch_size, args.block_len, args.code_rate_n),
                args,
                snr_low=args.train_dec_channel_low,
                snr_high=args.train_dec_channel_high,
                mode="decoder",
            )
            new_els = original_noise.numel()
            running_mean_original = (
                torch.mean(original_noise) * new_els
                + original_count * running_mean_original
            ) / (new_els + original_count)
            running_var_original = (
                torch.var(original_noise) * new_els
                + original_count * running_var_original
            ) / (new_els + original_count)
            sigma_original = torch.std(original_noise, dim=[1, 2])
            original_sigma_hist = (
                torch.histogram(sigma_original, bins=bins)[0] + original_sigma_hist
            )

        running_mean_new = 0
        running_var_new = 0
        new_count = 0
        new_sigma_hist = 0
        for i in trange(1000):
            new_noise = channel(enc_output) - enc_output
            new_els = new_noise.numel()
            running_mean_new = (
                torch.mean(new_noise) * new_els + new_count * running_mean_new
            ) / (new_els + new_count)
            running_var_new = (
                torch.var(new_noise) * new_els + new_count * running_var_new
            ) / (new_els + new_count)
            sigma_new = torch.std(new_noise, dim=[1, 2])
            new_sigma_hist = torch.histogram(sigma_new, bins=bins)[0] + new_sigma_hist

        torch.testing.assert_allclose(running_mean_original, running_mean_new)
        torch.testing.assert_allclose(running_var_original, running_var_new)
        torch.testing.assert_allclose(original_sigma_hist, new_sigma_hist)
        print("Passed!")

        print("Exiting")
        exit()

    if args.test_compare:
        # import sys
        # sys.path.append('/home/abhijeet/dev/turbo-codes/')
        # print(sys.path)

        print("Doing test of comparing decoders")

        import tensorflow as tf
        from old_src.turboae_adapters import (
            TFTurboAEDecoderCNN,
            TurboAEDecoderParameters,
        )

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        dec_to_save = model.dec
        state_dict = dec_to_save.state_dict()
        state_dict_path = (
            "./tmp/decoders/" + "_".join([identity, model_filename, "dec"]) + ".pt"
        )
        torch.save(dec_to_save.state_dict(), state_dict_path)
        torch_state_dict = torch.load(state_dict_path)
        params = TurboAEDecoderParameters.from_pytorch(torch_state_dict)
        tf_model = TFTurboAEDecoderCNN(params, block_len=100)

        batch_size = args.batch_size
        for i in range(10):
            test_input = (
                2
                * np.random.randint(0, 1, size=(batch_size, 100, 3)).astype(np.float32)
                - 1
                + np.random.normal(0, 1, size=(batch_size, 100, 3))
            )
            torch_test_input = torch.from_numpy(test_input)
            tf_test_input = tf.convert_to_tensor(test_input)

            torch_output = model.dec(torch_test_input)
            tf_output = tf.sigmoid(tf_model(tf_test_input))

            np.testing.assert_almost_equal(
                torch_output.cpu().detach().numpy(), tf_output.numpy(), decimal=5
            )
            print(f"{i} passed!")

        print("Exiting")
        exit()

    if args.test_compare_encoder_conversion:
        print(args.test_compare_encoder_conversion)

        print("Doing test of comparing converted encoder")
        # if not args.precompute_norm_stats:
        #     raise ValueError("Need to be using --precompute_norm_stats")

        import numpy as np
        import tensorflow as tf
        from old_src.turboae_adapters import (
            TFTurboAEEncoderCNN,
            TurboAEEncoderParameters,
        )

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        enc_to_save = model.enc
        state_dict = enc_to_save.state_dict()
        state_dict_path = (
            "./tmp/encoders/" + "_".join([identity, model_filename, "enc"]) + ".pt"
        )
        torch.save(enc_to_save.state_dict(), state_dict_path)
        torch_state_dict = torch.load(state_dict_path)
        params = TurboAEEncoderParameters.from_pytorch(torch_state_dict)
        tf_model = TFTurboAEEncoderCNN(params, block_len=100)

        batch_size = args.batch_size
        for i in range(100):
            test_input = np.random.randint(0, 1, size=(batch_size, 100, 1))
            torch_test_input = torch.from_numpy(test_input)
            tf_test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)

            # model.enc.num_test_block = 10000000
            # model.enc.mean_scalar = torch.tensor([-0.0215])
            # model.enc.std_scalar = torch.tensor([0.5114])
            torch_output = model.enc(torch_test_input)
            tf_output = tf_model(tf_test_input)

            np.testing.assert_almost_equal(
                torch_output.cpu().detach().numpy(), tf_output.numpy(), decimal=5
            )
            print(f"{i} passed!")

        print("Exiting")
        exit()

    if args.onnx_save_decoder:
        # torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        dec_to_save = model.dec
        torch.save(
            dec_to_save.state_dict(),
            "./tmp/decoders/" + "_".join([identity, model_filename, "dec"]) + ".pt",
        )
        np.save(
            "./tmp/decoders/" + "_".join([identity, "interleaver"]) + ".np", p_array1
        )
        # ONNX conversion
        dummy_input = torch.zeros((50, args.block_len, 3))
        input_names = ["received"]
        output_names = ["final"]
        torch.onnx.export(
            dec_to_save,
            dummy_input,
            "./tmp/decoders/" + "_".join([identity, model_filename, "dec"]) + ".onnx",
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                # dict value: manually named axes
                "received": {0: "batch"},
                "final": {0: "batch"},
            },
        )
        print("Exiting")
        exit()

    if args.test_onnx_decoder:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        import tensorflow.compat.v1 as tf1

        model_filename = os.path.splitext(os.path.basename(args.onnx_decoder_path))[0]
        onnx_dec = onnx.load(args.onnx_decoder_path)  # load onnx model

        onnx_tf_dec = prepare(onnx_dec)
        # graph_fname = './tmp/decoders/' + '_'.join([identity, model_filename])
        # onnx_tf_dec.export_graph(graph_fname)

        # INPUT_TENSOR_NAME = 'received.1:0'
        # OUTPUT_TENSOR_NAME = 'final:0'

        # with tf1.gfile.FastGFile(graph_fname + '/saved_model.pb', 'rb') as f:
        #     graph_def = tf1.GraphDef()
        #     graph_def.ParseFromString(f.read())

        # with tf1.Graph().as_default() as graph:
        #     tf1.import_graph_def(graph_def, name="")

        # input_tensor = graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        # output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

        torch_dec = model.dec

        for i in range(10):
            test_input = np.random.randint(0, 1, size=(75, 100, 3)).astype(np.float32)
            torch_test_input = torch.from_numpy(test_input)
            tf_test_input = tf.convert_to_tensor(test_input)

            torch_output = model.dec(torch_test_input)
            # with tf1.Session(graph=graph) as sess:
            #     tf_output = sess.run(output_tensor, feed_dict={input_tensor: tf_test_input})  #
            tf_output = onnx_tf_dec.run(tf_test_input)

            np.testing.assert_almost_equal(
                torch_output.cpu().detach().numpy(), tf_output.final, decimal=5
            )
            print(f"{i} passed!")

        print("Exiting")
        exit()

    if args.save_encoder:
        import os
        from pprint import pprint

        print("Saving encoder state dict from torch")

        model_filename = os.path.splitext(os.path.basename(args.init_nw_weight))[0]
        print(f"Model filename is {model_filename}")
        enc_to_save = model.enc
        state_dict = enc_to_save.state_dict()
        state_dict_path = (
            "./tmp/encoders/" + "_".join([identity, model_filename, "enc"]) + ".pt"
        )
        torch.save(enc_to_save.state_dict(), state_dict_path)

        print("Done saving")
        torch_state_dict = torch.load(state_dict_path)
        pprint(list(torch_state_dict.keys()))

        exit()

    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    if args.optimizer == "lookahead":
        print("Using Lookahead Optimizers")
        from optimizers import Lookahead

        lookahead_k = 5
        lookahead_alpha = 0.5
        if args.num_train_enc != 0 and args.encoder not in [
            "Turbo_rate3_lte",
            "Turbo_rate3_757",
        ]:  # no optimizer for encoder
            enc_base_opt = optim.Adam(model.enc.parameters(), lr=args.enc_lr)
            enc_optimizer = Lookahead(
                enc_base_opt, k=lookahead_k, alpha=lookahead_alpha
            )

        if args.num_train_dec != 0:
            dec_base_opt = optim.Adam(
                filter(lambda p: p.requires_grad, model.dec.parameters()),
                lr=args.dec_lr,
            )
            dec_optimizer = Lookahead(
                dec_base_opt, k=lookahead_k, alpha=lookahead_alpha
            )

        general_base_opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.dec_lr
        )
        general_optimizer = Lookahead(
            general_base_opt, k=lookahead_k, alpha=lookahead_alpha
        )

    else:  # Adam, SGD, etc....
        if args.optimizer == "adam":
            OPT = optim.Adam
        elif args.optimizer == "sgd":
            OPT = optim.SGD
        else:
            OPT = optim.Adam

        if args.num_train_enc != 0 and args.encoder not in [
            "Turbo_rate3_lte",
            "Turbo_rate3_757",
        ]:  # no optimizer for encoder
            enc_optimizer = OPT(model.enc.parameters(), lr=args.enc_lr)

        if args.num_train_dec != 0:
            dec_optimizer = OPT(
                filter(lambda p: p.requires_grad, model.dec.parameters()),
                lr=args.dec_lr,
            )

        general_optimizer = OPT(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.dec_lr
        )

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []

    for epoch in range(1, args.num_epoch + 1):

        if args.joint_train == 1 and args.encoder not in [
            "Turbo_rate3_lte",
            "Turbo_rate3_757",
        ]:
            for idx in range(args.num_train_enc + args.num_train_dec):
                train(
                    epoch,
                    model,
                    general_optimizer,
                    args,
                    use_cuda=use_cuda,
                    mode="encoder",
                )

        else:
            if args.num_train_enc > 0 and args.encoder not in [
                "Turbo_rate3_lte",
                "Turbo_rate3_757",
            ]:
                for idx in range(args.num_train_enc):
                    train(
                        epoch,
                        model,
                        enc_optimizer,
                        args,
                        use_cuda=use_cuda,
                        mode="encoder",
                    )

            if args.num_train_dec > 0:
                for idx in range(args.num_train_dec):
                    train(
                        epoch,
                        model,
                        dec_optimizer,
                        args,
                        use_cuda=use_cuda,
                        mode="decoder",
                    )

        this_loss, this_ber = validate(
            model, general_optimizer, args, use_cuda=use_cuda
        )
        report_loss.append(this_loss)
        report_ber.append(this_ber)

    if args.print_test_traj == True:
        print("test loss trajectory", report_loss)
        print("test ber trajectory", report_ber)
        print("total epoch", args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    torch.save(model.state_dict(), "./tmp/torch_model_" + identity + ".pt")
    print("saved model", "./tmp/torch_model_" + identity + ".pt")

    if args.is_variable_block_len:
        print("testing block length", args.block_len_low)
        test(model, args, block_len=args.block_len_low, use_cuda=use_cuda)
        print("testing block length", args.block_len)
        test(model, args, block_len=args.block_len, use_cuda=use_cuda)
        print("testing block length", args.block_len_high)
        test(model, args, block_len=args.block_len_high, use_cuda=use_cuda)

    else:
        test(model, args, use_cuda=use_cuda)
