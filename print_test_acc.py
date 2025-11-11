import re
import matplotlib.pyplot as plt

# Example file content
with open("log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-27_00-52-46/exp_log.txt", "r") as f:
    text = f.read()

# Example file content
with open("log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-29_15-54-06/exp_log.txt", "r") as f:
    text1 = f.read()


# Use regex to find all occurrences of numbers after "test_acc ="
test_acc_values = re.findall(r'test_acc\s*=\s*([\d.]+)', text)

test1_acc_values = re.findall(r'test_acc\s*=\s*([\d.]+)', text1)

# Convert to float list
test_acc_values = [float(x) for x in test_acc_values]

test1_acc_values = [float(x) for x in test1_acc_values]

#plt.figure(1)
#x = range(len(test_acc_values))
#plt.plot(x,test_acc_values,x,test1_acc_values)
#plt.show()

#%%

#rank-6

def getAccuracyList(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    test_acc_values = re.findall(r'test_acc\s*=\s*([\d.]+)', text)
    test_acc_values = [float(x) for x in test_acc_values]
    return test_acc_values



rank18 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-28_09-37-08/exp_log.txt');
rank32 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-28_01-40-45/exp_log.txt')
rank6 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-27_19-38-07/exp_log.txt')
rank12 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-27_09-54-13/exp_log.txt')
rank24 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/rank-24-fedhello/exp_log.txt')
rank48 = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-22_05-08-18/exp_log.txt')
rank48_fixA = getAccuracyList('log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-28_23-04-00/exp_log.txt')
rank48_truncate24 = getAccuracyList("log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-29_15-54-06/exp_log.txt")
rank48_fixA_svd = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50_2025-10-29_22-54-01/exp_log.txt')

proposed = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image-rank-var-b-only_2025-11-10_00-03-54/exp_log.txt')
rank_var = getAccuracyList('/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/depthffm_fim/image-rank-variation-only_2025-11-10_07-54-26/exp_log.txt')



x = range(len(rank6))
plt.plot(x, rank6,  label='Rank 6')
plt.plot(x, rank12, label='Rank 12')
plt.plot(x, rank18, label='Rank 18')
plt.plot(x, rank24, label='Rank 24')
#plt.plot(x, rank32, label='Rank 32')
#plt.plot(x, rank48, label='Rank 48')
#plt.plot(x,rank48_fixA,label='Rank 48 with fixed A')
#plt.plot(x,rank48_truncate24,label='Rank 48 truncate to 24')
#plt.plot(x,rank48_fixA_svd ,label='Rank48 fixed A + SVD')
plt.plot(x, proposed, label='proposed - budget 24')
plt.plot(x, rank_var, label='rank var -  budget 24')

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Round for Different Ranks")
plt.legend()          # <-- shows labels
plt.grid(True)
plt.show()



