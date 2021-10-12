from argparse import ArgumentParser
from src.image_producer.abstract_image_producer import ExtensionType
from typing import Any
from src import image_producer
from src.image_producer.super_resolution_image_producer import SuperResolutionImageProducer
from src.image_producer.simple_image_producer import SimpleImageProducer


def add_image_producer_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("Image Producer Options")
    parser.add_argument("--output_method", help="The method used for Image Production. Super resolution results in a higher resolution output but takes significantly longer. Options: ['simple', 'super_resolution'] (default: %(default)s)", type=str, default='simple')
    parser.add_argument("--extension", help="File extension to save images as. Options ['PNG', 'JPEG']. (default: %(default)s)", type=str, default='PNG')
    parser.add_argument("--super_res_up_factor", help='For: --output_method super_resolution, otherwise ignored. Upscaling factor used to upscale image. (default: %(default)s)', default=4.0, type=float)
    parser.add_argument("--super_res_samples", help='For: --output_method super_resolution, otherwise ignored. Number of samples to generate for stacking super resolution image. (default: %(default)s)', default=16, type=int)
    parser.add_argument("--super_res_translate_range", help='For: --output_method super_resolution, otherwise ignored. The range for translations. Example of 5.0 will produce translations between -5.0 and 5.0. (default: %(default)s)', default=8., type=float)
    parser.add_argument("--super_res_margin_crop", help='For: --output_method super_resolution, otherwise ignored. Crops margin off the edges of super resolution image to avoid displaying the layered images. Results in slight loss of image resolution. (default: %(default)s)', default=True, type=bool)
    return parent_parser


def get_image_producer(model, **kwargs: Any):

    extension_type = None

    if kwargs['extension'].upper() == 'PNG':
        extension_type = ExtensionType.PNG
    elif kwargs['extension'].upper() == 'JPEG':
        extension_type = ExtensionType.JPEG
    else:
        raise Exception(f"Invalid Extension Type! {kwargs['extension']} is not an option. Options: ['PNG', 'JPEG'].")

    if kwargs['output_method'] == 'simple':
        
        image_producer = SimpleImageProducer(model, extension_type)
        image_producer.set_to_eval()
        return image_producer

    elif kwargs['output_method'] == 'super_resolution':

        image_producer = SuperResolutionImageProducer(model, extension_type, kwargs['super_res_up_factor'], kwargs['super_res_samples'], kwargs['super_res_translate_range'], kwargs['super_res_margin_crop'])
        image_producer.set_to_eval()
        return image_producer

    else:
        
        raise Exception(f"Unknown --output_method! {kwargs['output_method']} is not a supported option, the options are ['simple', 'super_resolution'].")
