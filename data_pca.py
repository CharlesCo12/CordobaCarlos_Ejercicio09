import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#Para los Arrestos
datos=pd.read_csv('USArrests.csv')
tit=['Murder','Assault','UrbanPop','Rape']
dat=datos[tit]
dat=(dat-dat.mean())/dat.std()
cov=np.cov(dat.T)
eigval,eigvec = np.linalg.eig(cov)

#Cálculo de la Varianza de los arrestos
plt.figure()
porc=[]
for i in range(4):
    porc.append(np.sum(eigval[:i+1])*25)
x1=np.arange(1,5)
plt.plot(x1,porc)
plt.scatter(x1,porc)
plt.grid()
plt.ylim(0,100)
plt.xlabel('Número de Autovalores')
plt.ylabel('Porcentaje de Varianza')
plt.savefig('varianza_arrestos.png')
plt.close()

#Componentes principales de los arrestos
v1=eigvec[:,0]
v2=eigvec[:,1]

plt.figure(figsize=(12,12))
for i in range(len(datos)):
    ciudad = datos['Unnamed: 0'][i]
    v = np.array(dat.iloc[i])
    x = np.dot(v1, v) 
    y = np.dot(v2, v) 
    plt.text(x,y, ciudad, fontsize=10, color='blue')
    plt.scatter(x, y, s=0.001)
    
for j in range(len(tit)):
    plt.arrow(0.0, 0.0, 3*v1[j], 3*v2[j], color='red', head_width=0.1)
    plt.text(3.2*v1[j], 3.2*v2[j], tit[j], color='red')

plt.ylim(-3,3)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('arrestos.png')
plt.close()


#Para los Carros
datos_car=pd.read_csv('Cars93.csv')
tit_car=['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile','Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
dat_carros=datos_car[tit_car]
dat_carros=(dat_carros-dat_carros.mean())/dat_carros.std()
cov_car=np.cov(dat_carros.T)
eigval_car,eigvec_car = np.linalg.eig(cov_car)

#Cálculo de la Varianza de los Carros
plt.figure()
porc_car=[]
for i in range(11):
    porc_car.append(np.sum(eigval_car[:i+1])*10)
x1_car=np.arange(1,12)
plt.plot(x1_car,porc_car)
plt.scatter(x1_car,porc_car)
plt.grid()
plt.ylim(0,100)
plt.xlabel('Número de Autovectores')
plt.ylabel('Porcentaje de la Varianza')
plt.savefig('varianza_cars.png')
plt.close()

#Componentes principales de los carros
v1_car=eigvec_car[:,0]
v2_car=eigvec_car[:,1]

plt.figure(figsize=(12,12))
for i in range(len(datos_car)):
    carro = datos_car['Model'][i]
    v = np.array(dat_carros.iloc[i])
    x = np.dot(v1_car, v) 
    y = np.dot(v2_car, v) 
    plt.text(x,y, carro, fontsize=10, color='blue')
    plt.scatter(x, y, s=0.001)
    
for j in range(len(tit_car)):
    plt.arrow(0.0, 0.0, 5*v1_car[j], 5*v2_car[j], color='red', head_width=0.1)
    plt.text(5*v1_car[j], 5*v2_car[j], tit_car[j], color='red')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('cars.png')
plt.close()