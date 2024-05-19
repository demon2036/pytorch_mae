import yaml

# def read_yaml(yaml_file_path: str = 'configs/mod_custom/tiny.yaml'):
#     print_with_seperator(f'Read yaml From {yaml_file_path}')
#     with open(yaml_file_path, 'r') as f:
#         yaml_data = yaml.safe_load(f)
#         json_print(yaml_data)
#         print()
#     return yaml_data
#
#
# def get_config(args):
#     args_dict = {key: value for key, value in vars(args).items() if value is not None and key != 'yaml_path'}
#     print('Using Default Config From Yaml')
#     yaml_data = read_yaml(args.yaml_path)
#
#     if len(args_dict) > 0:
#         print_with_seperator("Using Custom Setting")
#         json_print(args_dict)
#         yaml_data.update(args_dict)
#         print_with_seperator('Now Setting')
#         json_print(yaml_data)
#     return yaml_data


with open('configs/mae/adv_feat/tiny.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)
    print(yaml_data)
