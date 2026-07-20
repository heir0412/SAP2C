import os
import ntpath
import tqdm
import torch
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from torchvision import transforms

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.sync_batchnorm import convert_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from util.flow_to_color import flow2img
from util.util import load_image, save_image
from transformers import CLIPTokenizer


DEFAULT_SINGLE_PROMPT = (
    "Subtle but visible natural motion, with gentle flowing water, soft ripples, "
    "slow drifting waves, coherent local direction, stable temporal consistency, "
    "realistic deformation, and minimal flicker. The motion should feel alive "
    "but not violent or overly turbulent."
)


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)

    # ======= 新增：只做单图推理 =======
    if opt.inference_only:
        assert opt.single_image and os.path.isfile(opt.single_image), \
            f'--single_image not found: {opt.single_image}'
        assert opt.single_mask and os.path.isfile(opt.single_mask), \
            f'--single_mask not found: {opt.single_mask}'
        if opt.use_prompts:
            if not opt.single_prompt:
                opt.single_prompt = DEFAULT_SINGLE_PROMPT
                print(f'[inference_only] Using built-in prompt: {opt.single_prompt}')
        elif opt.netG == "spadexattnunetsd":
            raise RuntimeError(
                "netG=spadexattnunetsd is prompt-conditioned and requires "
                "--use_prompts. If --single_prompt is omitted, test_motion.py "
                "uses a built-in generic motion prompt."
            )

        # 创建模型
        model = create_model(opt)
        if 'sync' in opt.norm:
            model = convert_model(model)
        model.eval()

        # 读图 [1,3,H,W]
        img = load_image(opt.single_image,
                         target_width=opt.fineSize,
                         target_height=opt.fineSize).unsqueeze(0)

        # 读 mask [1,1,H,W]
        m = Image.open(opt.single_mask).convert('L').resize(
                (opt.fineSize, opt.fineSize), Image.NEAREST)
        m = transforms.functional.to_tensor(m).unsqueeze(0)

        # 拼 RGBA [1,4,H,W]
        label = torch.cat([img, m], dim=1)

        # 文本 prompt
        if opt.use_prompts:
            tokenizer = CLIPTokenizer.from_pretrained(opt.tokenizer_path)
            prompt_ids = tokenizer(
                opt.single_prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            with torch.no_grad():
                gen = model.inference(label, label, label, prompt_ids)
        else:
            with torch.no_grad():
                gen = model.inference(label, label, label)

        # resize & scale
        outSize = opt.out_size
        scale = [outSize / gen.shape[3], outSize / gen.shape[2]]
        gen = gen * torch.tensor(scale, dtype=gen.dtype, device=gen.device).view(1, 2, 1, 1)
        gen = F.interpolate(gen, (outSize, outSize))

        # 按 mask 取出光流（可注释掉保留全图）
        m_up = F.interpolate(m, (outSize, outSize))
        gen_masked = gen.cpu() * m_up.cpu()

        # ======= 保存到输入图像同级目录（带输入图名） =======
        input_dir = os.path.dirname(opt.single_image)
        base_name = os.path.splitext(os.path.basename(opt.single_image))[0]  # 如 00001_realistic

        flow_tensor_path = os.path.join(input_dir, f'{base_name}_Motion.pt')
        flow_vis_path = os.path.join(input_dir, f'{base_name}_flow_masked.jpg')

        torch.save(gen_masked, flow_tensor_path)
        vis = flow2img(gen_masked[0])
        save_image(vis, flow_vis_path)

        print(f'[inference_only] Saved flow tensor: {flow_tensor_path}')
        print(f'[inference_only] Saved visualization: {flow_vis_path}')
        raise SystemExit(0)

    # ======= 原始批量评估流程 =======
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s'
                        % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)
    if 'sync' in opt.norm:
        model = convert_model(model)
    model.eval()

    for i, data in tqdm.tqdm(enumerate(dataset)):
        if i >= opt.how_many:
            break

        if opt.use_prompts:
            generated = model.inference(data['label'], data['inst'],
                                        data['image'], data['prompt'])
        else:
            generated = model.inference(data['label'], data['inst'],
                                        data['image'])

        if data['label'][0].shape[0] == 4:
            mask = (data['label'][0][3:4, :, :] == 1.).float()
            generated = generated.detach().cpu()
            generated_masked = (generated.clone() * mask) + \
                               (torch.zeros_like(generated) * (1. - mask))
            visuals = OrderedDict([
                ('input_image', util.tensor2im(data['label'][0][:3, :, :])),
                ('mask', util.tensor2im(mask, normalize=False)),
                ('synthesized_flow_masked', flow2img(generated_masked.data[0])),
                ('synthesized_flow', flow2img(generated.data[0])),
                ('real_flow', flow2img(data['image'][0]))
            ])
        else:
            visuals = OrderedDict([
                ('input_image', util.tensor2im(data['label'][0])),
                ('synthesized_flow', flow2img(generated.data[0])),
                ('real_flow', flow2img(data['image'][0]))
            ])

        img_path = data['path'][0]
        save_dir = webpage.get_image_dir()
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        flow_name = '%s_%s.pt' % (name.replace('_input', ''), 'Motion')
        save_path = os.path.join(save_dir, flow_name)
        torch.save(generated.data.detach().cpu(), save_path)

        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
