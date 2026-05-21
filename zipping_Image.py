import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img=np.array(Image.open('MemePrint-Logo-1-1.png').convert('L'))

U,S,Vt=np.linalg.svd(img,full_matrices=False)
def reconstruct_image(U,S,Vt,k):
  return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
k_values=[8,25,1000]
errors=[]
variance_retained=[]
m,n=img.shape
total_variance=np.sum(S**2)

for k in k_values:
  img_reconstructed=reconstruct_image(U,S,Vt,k)
  error=np.linalg.norm(img - img_reconstructed, ord='fro')
  errors.append(error)
  variance=np.sum(S[:k]**2)/total_variance
  variance_retained.append(variance)

  compression_ratio=(k*(m+n+1))/(m*n)
  print(f'k={k}: Sai so Frobenius={error:.3f}, phuong sai giu lai={variance:.3f}, Ty le nen={compression_ratio:.3f}')

plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.title('Anh goc')
plt.axis('off')

for i,k in enumerate(k_values):
  img_reconstructed=reconstruct_image(U,S,Vt,k)
  plt.subplot(1,4,i+2)
  plt.imshow(img_reconstructed,cmap='gray')
  plt.title(f'k={k}')
  plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(S,marker='o',color='#1f77b4')
plt.title('Gia tri singular')
plt.xlabel('Chi so')
plt.ylabel('Gia tri singular')

plt.subplot(1,2,2)
plt.plot(k_values,errors,marker='o',label='Sai so Frobenius', color='#ff7f0e')
plt.plot(k_values,variance_retained,marker='s',label='Ty le phuong sai giu lai', color='#2ca02c')
plt.title('Sai so va phuong sai theo k')
plt.xlabel('k')
plt.ylabel('Gia tri')
plt.legend()
plt.tight_layout()
plt.show()
