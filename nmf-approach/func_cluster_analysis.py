
cluster =  dict()
count = 0

# with open('email-v1005-e25571-c42/email-Eu-core-department-labels.txt') as f:
with open('email-v1005-e25571-c42/output.txt') as f:	 # output
    for line in f:
    	(key, val) = line.split()
    	if val not in cluster: 
    		# print(val)
    		cluster[val] = [key]
    		count += 1
    		# print(cluster[val])
    	else:
    		cluster[val].append(key)

print("count", count)
a = cluster.get(str(41))
# print(a)
print(len(a))