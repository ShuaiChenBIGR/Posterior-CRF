
from Training_lrate_generator import train_model
import Modules.Common_modules as cm


data_seed_list = [
                  1,
                  2,
                  3,
                  4,
                  5,
                  ]

job_list = [
            'CNN_baseline',
            # 'Intensity_CRF',
            # 'Spatial_CRF',
            # 'Posterior_CRF',
            # 'Posterior_CRF_pretrain',
            # 'Post_CRF'
            ]

data_split_list = ['36L']


Test_only = False

num_epoch = 200

folder_name = 'results/'

test_results = 0

# Run jobs:
for job in job_list:
    if job == 'CNN_baseline':
        split = '36L'
        Test_wmh_results = [0, 0, 0, 0, 0]

        for seed in data_seed_list:
            test_results = train_model(Test_only, job, split, seed, folder_name, num_epoch)

            ####################   test Aorta each seed
            for i, item in enumerate(test_wmh_results):
                Test_wmh_results[i] += item

            file = open(
                str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/test_WMH_seed_' + str(
                    seed) + '.txt', 'w')
            title = 'Dice          H95           AVD               Recall             F1\n'
            history = (
                '{:4f}        {:.4f}         {:.4f}        {:.4f}        {:.4f}\n'
                    .format(test_results[0], test_results[1], test_results[2], test_results[3], test_results[4]))
            file.write(title)
            file.write(history)
            file.close()

        #################### Test overall
        file = open(str(folder_name) + str(job) + '/' + str(split) + '/test_overall.txt', 'w')
        title = 'Dice          H95           AVD               Recall             F1\n'
        history1 = (
            '{:4f}        {:.4f}         {:.4f}        {:.4f}        {:.4f}\n'
                .format(test_results[0] / len(data_seed_list), test_results[1] / len(data_seed_list),
                        test_results[2] / len(data_seed_list), test_results[3] / len(data_seed_list),
                        test_results[4] / len(data_seed_list)))

        file.write(title)
        file.write(history1)
        file.close()

    else:
        for split in data_split_list[:]:
            test_results = [0, 0, 0, 0, 0]
            for seed in data_seed_list:
                test_results = train_model(Test_only, job, split, seed, folder_name, num_epoch)

                ####################   test WMH each seed
                for i, item in enumerate(test_results):
                    test_results[i] += item

                file = open(
                    str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/test_WMH_seed_' + str(
                        seed) + '.txt', 'w')
                title = 'Dice          H95           AVD               Recall             F1\n'
                history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}        {:.4f}\n'
                        .format(test_results[0], test_results[1], test_results[2], test_results[3],
                                test_results[4]))
                file.write(title)
                file.write(history)
                file.close()


            #################### Test overall
            file = open(str(folder_name) + str(job) + '/' + str(split) + '/test_overall.txt', 'w')
            title = 'Dice          H95           AVD               Recall             F1\n'
            history1 = (
                '{:4f}        {:.4f}         {:.4f}        {:.4f}        {:.4f}\n'
                    .format(test_results[0] / len(data_seed_list), test_results[1] / len(data_seed_list),
                            test_results[2] / len(data_seed_list), test_results[3] / len(data_seed_list),
                            test_results[4] / len(data_seed_list)))
            file.write(title)
            file.write(history1)
            file.close()

print('All jobs finished')
