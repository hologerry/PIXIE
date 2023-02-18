import argparse
import io
import os
import pickle
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from zipreader import ZipReader

from pixielib.datasets.body_datasets import TestData
from pixielib.pixie import PIXIE
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils.tensor_cropper import transform_points
from pixielib.visualizer import Visualizer


def load_data(args):

    split_meta_file = os.path.join(args.data_folder, f"{args.split}.pkl")
    with open(split_meta_file, "rb") as f:
        split_meta_dicts = pickle.load(f)
    # keys: keys(['seq_len', 'img_dir', 'name', 'video_file', 'label'])
    return split_meta_dicts


def read_img(frame_path):
    img_data = ZipReader.read(frame_path)
    rgb_im = Image.open(io.BytesIO(img_data)).convert("RGB")
    return rgb_im


def prepare_image_folder(args, vid_dict):
    video_name = vid_dict["name"]
    seq_len = vid_dict["seq_len"]
    img_dir = vid_dict["img_dir"]
    frames_zip_file = args.frames_zip_file

    image_folder = os.path.join("/tmp/MSASL", video_name)
    os.makedirs(image_folder, exist_ok=True)
    frame_paths = [f"{frames_zip_file}@{img_dir}{frame_id:04d}.png" for frame_id in range(seq_len)]

    frame_images = [read_img(frame_path) for frame_path in frame_paths]
    for frame_id, frame_image in enumerate(frame_images):
        frame_image.save(os.path.join(image_folder, f"{frame_id:04d}.png"))
    orig_width, orig_height = frame_images[0].size
    return video_name, image_folder


def images_to_video(img_folder, img_pose_fix, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-threads",
        "16",
        "-i",
        f"{img_folder}/%04d{img_pose_fix}",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]

    # print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def main(args):
    assert args.job_idx < args.job_num, f"job_idx {args.job_idx} should be smaller than job_num {args.job_num}"
    assert args.gpu_idx < args.gpu_num, f"gpu_idx {args.gpu_idx} should be smaller than gpu_num {args.gpu_num}"

    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    # check env
    if not torch.cuda.is_available():
        print("CUDA is not available! use CPU instead")
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # -- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(
        render_size=args.render_size, config=pixie_cfg, device=device, rasterizer_type=args.rasterizer_type
    )
    use_deca = False

    # -- load data
    cur_split_meta_dicts = load_data(args)
    print(f"split {args.split} contains {len(cur_split_meta_dicts)} videos.")

    cur_job_gpu_idx = args.gpu_idx + args.gpu_num * args.job_idx
    num_total_gpus = args.gpu_num * args.job_num
    cur_job_gpu_split_meta_dicts = cur_split_meta_dicts[cur_job_gpu_idx::num_total_gpus]

    for vid_dict in tqdm(cur_job_gpu_split_meta_dicts):
        video_name, image_folder = prepare_image_folder(args, vid_dict)

        output_pkl_file = os.path.join(args.savefolder, "pixie_outs", f"{video_name}_param_and_pred.pkl")
        if os.path.exists(output_pkl_file):
            print(f"skipping {video_name}")
            continue

        testdata = TestData(image_folder, iscrop=args.iscrop, body_detector="rcnn")

        cur_video_vis_folder = f"{image_folder}_output"
        os.makedirs(cur_video_vis_folder, exist_ok=True)

        cur_video_param_and_pred = {}
        cur_video_param_dicts = {}
        cur_video_pred_dicts = {}

        for frame_i, batch in enumerate(testdata):
            util.move_dict_to_device(batch, device)
            batch["image"] = batch["image"].unsqueeze(0)
            batch["image_hd"] = batch["image_hd"].unsqueeze(0)

            # name = batch["name"]
            # print(name)
            # frame_id = int(name.split('frame')[-1])
            # name = f'{frame_id:05}'

            data = {"body": batch}
            param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
            # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
            # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
            moderator_weight = param_dict["moderator_weight"]
            codedict = param_dict["body"]
            opdict = pixie.decode(codedict, param_type="body")
            opdict["albedo"] = visualizer.tex_flame2smplx(opdict["albedo"])
            # if args.saveObj or args.saveParam or args.savePred or args.saveImages or args.deca_path is not None:
            #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            # -- save results
            # run deca if deca is available and moderator thinks information from face crops is reliable
            # if args.deca_path is not None and param_dict["moderator_weight"]["head"][0, 1].item() > 0.6:
            #     cropped_face_savepath = os.path.join(savefolder, name, f"{name}_facecrop.jpg")
            #     cv2.imwrite(cropped_face_savepath, util.tensor2image(data["body"]["head_image"][0]))
            #     _, deca_opdict, _ = deca.run(cropped_face_savepath)
            #     flame_displacement_map = deca_opdict["displacement_map"]
            #     opdict["displacement_map"] = visualizer.tex_flame2smplx(flame_displacement_map)
            # if args.lightTex:
            #     visualizer.light_albedo(opdict)
            # if args.extractTex:
            #     visualizer.extract_texture(opdict, data["body"]["image_hd"])
            # if args.reproject_mesh and args.rasterizer_type == "standard":
            #     ## whether to reproject mesh to original image space
            #     tform = batch["tform"][None, ...]
            #     tform = torch.inverse(tform).transpose(1, 2)
            #     original_image = batch["original_image"][None, ...]
            #     visualizer.recover_position(opdict, batch, tform, original_image)
            if args.saveVis:
                if args.showWeight is False:
                    moderator_weight = None
                visdict = visualizer.render_results(
                    opdict,
                    data["body"]["image_hd"],
                    overlay=True,
                    moderator_weight=moderator_weight,
                    use_deca=use_deca,
                )
                # show cropped parts
                # if args.showParts:
                #     visdict["head"] = data["body"]["head_image"]
                #     visdict["left_hand"] = data["body"]["left_hand_image"]  # should be flipped
                #     visdict["right_hand"] = data["body"]["right_hand_image"]
                cv2.imwrite(
                    os.path.join(cur_video_vis_folder, f"{frame_i:04d}_vis.jpg"),
                    visualizer.visualize_grid(visdict, size=args.render_size),
                )
                # print(os.path.join(savefolder, f'{name}_vis.jpg'))
                # import ipdb; ipdb.set_trace()
                # exit()
            # if args.saveGif:
            #     visualizer.rotate_results(opdict, visdict=visdict, savepath=os.path.join(savefolder, f"{name}_vis.gif"))
            # if args.saveObj:
            #     visualizer.save_obj(os.path.join(savefolder, name, f"{name}.obj"), opdict)
            # if args.saveParam:
            #     codedict["bbox"] = batch["bbox"]
            #     util.save_pkl(os.path.join(savefolder, name, f"{name}_param.pkl"), codedict)
            #     np.savetxt(os.path.join(savefolder, name, f"{name}_bbox.txt"), batch["bbox"].squeeze())
            # if args.savePred:
            #     util.save_pkl(os.path.join(savefolder, name, f"{name}_prediction.pkl"), opdict)
            # if args.saveImages:
            #     for vis_name in visdict.keys():
            #         cv2.imwrite(
            #             os.path.join(savefolder, name, f"{name}_{vis_name}.jpg"), util.tensor2image(visdict[vis_name][0])
            #         )
            codedict["bbox"] = batch["bbox"]
            # cur_video_param_and_pred[f"{frame_i:04d}_param"] = codedict
            # cur_video_param_and_pred[f"{frame_i:04d}_pred"] = opdict
            for k, v in codedict.items():
                if k not in cur_video_param_dicts:
                    cur_video_param_dicts[k] = []
                if torch.is_tensor(v):
                    v = v[0].detach().cpu().numpy()
                cur_video_param_dicts[k].append(v)

            for k, v in opdict.items():
                if k not in cur_video_pred_dicts:
                    cur_video_pred_dicts[k] = []
                if torch.is_tensor(v):
                    v = v[0].detach().cpu().numpy()
                cur_video_pred_dicts[k].append(v)

        if args.saveVis:
            save_name = f"{video_name}.mp4"
            save_name = os.path.join(args.savefolder, "pixie_videos", save_name)
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            # print(f"Saving result video to {save_name}")
            images_to_video(img_folder=cur_video_vis_folder, img_pose_fix="_vis.jpg", output_vid_file=save_name)
            shutil.rmtree(cur_video_vis_folder)

        shutil.rmtree(image_folder)

        for k, v in cur_video_param_dicts.items():
            cur_video_param_and_pred[k] = np.stack(v, axis=0)

        for k, v in cur_video_pred_dicts.items():
            cur_video_param_and_pred[k] = np.stack(v, axis=0)

        os.makedirs(os.path.join(args.savefolder, "pixie_outs"), exist_ok=True)
        util.save_pkl(output_pkl_file, cur_video_param_and_pred)

    print(f"Finished! Please check the results in {savefolder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIXIE")

    parser.add_argument(
        "-i",
        "--inputpath",
        default="TestSamples/sign",
        type=str,
        help="path to the test data, can be image folder, image path, image path list, video",
    )
    parser.add_argument(
        "-s",
        "--savefolder",
        default="../PIXIE_MSASL_results",
        type=str,
        help="path to the output directory, where results(obj, txt files) will be stored.",
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="set device, cpu for using cpu")
    # process test images
    parser.add_argument(
        "--iscrop",
        default=True,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to crop input image, set false only when the test image are well cropped",
    )
    # rendering option
    parser.add_argument("--render_size", default=1024, type=int, help="image size of renderings")
    parser.add_argument(
        "--rasterizer_type", default="pytorch3d", type=str, help="rasterizer type: pytorch3d or standard"
    )
    parser.add_argument(
        "--reproject_mesh",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to reproject the mesh and render it in original image space, \
                            currently only available if rasterizer_type is standard, will add supports for pytorch3d \
                            after pytorch **stable version** supports non-squared images. \
                            default is False, means using the cropped image and its corresponding results",
    )
    # texture options
    parser.add_argument(
        "--deca_path",
        # default="../DECA",
        default=None,
        type=str,
        help="absolute path of DECA folder, if exists, will return facial details by running DECA. \
                        please refer to https://github.com/YadiraF/DECA",
    )
    parser.add_argument(
        "--useTex",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model",
    )
    parser.add_argument(
        "--lightTex",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to return lit albedo: that add estimated SH lighting to albedo",
    )
    parser.add_argument(
        "--extractTex",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to extract texture from input image, only do this when the face is near frontal and very clean!",
    )
    # save
    parser.add_argument(
        "--saveVis",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to save visualization of output",
    )
    parser.add_argument(
        "--showParts",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to show head/hands crops in visualization",
    )
    parser.add_argument(
        "--showWeight",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to visualize the moderator weight on colored shape",
    )
    parser.add_argument(
        "--saveGif",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to visualize other views of the output, save as gif",
    )
    parser.add_argument(
        "--saveObj",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to save outputs as .obj, \
                            Note that saving objs could be slow",
    )
    parser.add_argument(
        "--saveParam",
        default=True,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to save parameters as pkl file",
    )
    parser.add_argument(
        "--savePred",
        default=True,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to save smplx prediction as pkl file",
    )
    parser.add_argument(
        "--saveImages",
        default=False,
        type=lambda x: x.lower() in ["true", "1"],
        help="whether to save visualization output as separate images",
    )

    parser.add_argument(
        "--frames_zip_file",
        type=str,
        default="../data/MSASL/msasl_frames1.zip",
        help="input frames zip file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="split of the dataset to run the demo on.",
    )
    parser.add_argument("--data_folder", type=str, default="../data/MSASL/")

    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=10)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)

    main(parser.parse_args())
