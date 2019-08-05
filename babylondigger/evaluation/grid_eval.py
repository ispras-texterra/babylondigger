import json
import copy
from babylondigger.evaluation import evaluation as evl
import itertools as it


def __merge_dicts(hyper_dict, grid_dict):
    for key in grid_dict:
        if key in hyper_dict:
            if isinstance(hyper_dict[key], dict) and isinstance(grid_dict[key], dict):
                __merge_dicts(hyper_dict[key], grid_dict[key])
            else:
                hyper_dict[key] = grid_dict[key]
        else:
            raise ValueError("Undefined parameter: {}".format(key))
    return hyper_dict


def __parse_arguments_with_grid():
    parser = evl._build_parser()
    parser.add_argument('--grid', dest='grid', help='path to json file')
    args = parser.parse_args()
    with open(args.grid) as f:
        grid = json.load(f)
    return evl._parse_arguments(parser), grid


def __build_list_of_new_hyperparams(grid_params, params):
    params_list = []
    for grid_element in grid_params:
        hypers = copy.deepcopy(params)
        for grid_dict in grid_element.values():
            hypers = __merge_dicts(hypers, grid_dict)
        params_list.append(hypers)
    return params_list


def __take_all_key_path(dictt, new_list):
    for key in dictt:
        saved_dict = dictt
        if isinstance(dictt[key], dict):
            new_list.append(key)
            return __take_all_key_path(dictt[key], new_list)
        new_list.append(list(saved_dict.keys())[0])
        return tuple([elem for elem in new_list])


def __check_is_grid_valid(grid):
    new_set = set()
    big_set = set()
    for key in grid:
        for dictt in grid[key]:
            new_set.add(__take_all_key_path(dictt, []))
        if len(new_set) != 1:
            raise ValueError("Too many types of parameters in one grid key! Only one is allowed.")
        big_set = big_set.union(new_set)
        new_set.clear()
    if len(big_set) != len(grid):
        raise ValueError("Grid has intersections! Different params shouldn't overlap.")


def main():
    args, grid = __parse_arguments_with_grid()

    __check_is_grid_valid(grid)
    grid_params = [dict(zip(grid, prod)) for prod in it.product(*(grid[var_name] for var_name in grid))]

    params_list = __build_list_of_new_hyperparams(grid_params, args.hyperparams)

    evl.main(evl._Arguments(args.settings, args.method, params_list, delete_after=True))


if __name__ == "__main__":
    main()


