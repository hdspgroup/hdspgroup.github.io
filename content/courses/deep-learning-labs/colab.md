---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: Google Colaboratory
linktitle: Introducción
toc: true
type: docs
date: "2020-06-29T00:00:00+01:00"
draft: false
weight: 2
menu:
  deep-learning-labs:
    parent: Google Colaboratory
    weight: 1
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosh93/carlosh93.github.io/blob/master/files/introduccion_colab.ipynb)
<!--<a href="https://colab.research.google.com/github/carlosh93/carlosh93.github.io/blob/master/files/introduccion_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>-->


Antes de empezar, da click en el boton "Open in Colab" de arriba. Esto abrirá este notebook de python directamente en Colab. Luego, para poder editar y ejecutar el código, deberas copiar el notebook en Google drive utilizando el boton:


![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/copy_drive.png)

De esta manera, este notebook quedará almacenado en Google drive, en la carpeta *Colab Notebooks*

## Introducción

El objetivo de este tutorial es proporcionar un flujo de trabajo para entrenar modelos de Deep Learning. El flujo de trabajo propuesto requiere tener instalado los siguientes componentes en nuestra computadora:

1.   Cliente de sincronización de Google Drive.
2.   Editor de texto o IDE de python como Pycharm.

**La idea general será tener nuestro código python en una carpeta en Google Drive. De esta manera, utilizaremos el cliente de sincronización para modificar los archivos de manera local en nuestras computadoras (utilizando el editor de texto o un IDE de python) y que los cambios se vean reflejados directamente en Colab.** Aunque Colab permite editar archivos (simplemente dando doble click en el archivo deseado en el menu de la izquierda) es mucho mas sencillo y cómodo editar y manipular archivos complejos en un IDE de python.

### Cliente Google Drive
Para sistemas operativos Windows y Mac el cliente de Google drive puede ser descargado directamente de la pagina de google drive

https://www.google.com/drive/download/

Sin embargo, para sistemas operativos Linux no existe un cliente oficial de google drive, pero existen alternativas excelentes como Insync (Software pago una única vez) o el cliente para sistemas Ubuntu:

https://cambiatealinux.com/instalar-google-drive-en-ubuntu

### Editor de Python
Para trabajar con proyectos Python, es recomendable utilizar el entorno de desarrollo integrado (IDE) Pycharm. Actualmente, es posible acceder a una licencia de estudiante de la version profesional de Pycharm por un año:

https://www.jetbrains.com/es-es/community/education/#students

Para acceder al beneficio solo se necesita el correo electronico institucional (@correo.uis.edu.co o @saber.uis.edu.co o @uis.edu.co).

Si no se desea utilizar un IDE, es posible trabajar con un editor avanzado de texto: Visual Studio Code:

https://code.visualstudio.com/


Note que, debido a que nuestra intención es solo modificar el código en nuestra computadora local y ejecutar el código en Colab, no necesitamos tener instalado Python y las librerias de Tensorflow en nuestra computadora ya que estas se encuentran instaladas en Colab. Sin embargo, es recomendable instalar Python (Anaconda) y configurarlo con Pycharm en nuestra computadora local. A continuación proporciono links de interes:


1.   Instalar Anaconda: Windows: https://docs.anaconda.com/anaconda/install/windows/, Linux: https://docs.anaconda.com/anaconda/install/linux/, Mac: https://docs.anaconda.com/anaconda/install/mac-os/

2.   Configurar Pycharm con Anaconda: https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/

3.   Instalar liberias de Deep Learning: https://asociacionaepi.es/primeros-pasos-con-tensorflow/



## Solicitar Recursos a Colab
Una vez creemos un notebook en Colab, para solicitar recursos debemos primero seleccionar el entorno de ejecución adecuado y dar click en el botón Conectar:

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/sel_entorno_conect.gif)

Una vez conectados, Colab nos asigna un entorno **Linux con 25.51 GB de Ram, 68 GB de disco duro** y una GPU cuyas caracteristicas podemos consultar ejecutando la siguiente celda

### Características de la GPU asignada




```
# Check nvidia and nvcc cuda compiler

!nvidia-smi
!/usr/local/cuda/bin/nvcc --version
```

    Tue May 19 00:16:53 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.82       Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   34C    P0    26W / 250W |      0MiB / 16280MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Sun_Jul_28_19:07:16_PDT_2019
    Cuda compilation tools, release 10.1, V10.1.243


Por lo general, Colab asigna de manera aleatoria GPUs con diferente cantidad de memoria. Es recomendable utilizar una GPU de 16 o 15 GB de memoria para proyectos que requieran mucha memoria o trabajen con muchos datos. En caso de que Colab no nos asigne una GPU adecuada (algunas veces asigna GPUs de 8 o 11 GB de memoria) para nuestro proyecto, siempre podemos realizar distintas solicitudes hasta conseguir una GPU adecuada. Para esto debemos cerrar la sesion actual y volver a solicitar recursos:

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/get_new_GPU.gif)

Este proceso se repite varias veces hasta que se nos asigne la GPU deseada. Note que es posible que requiera recargar la ventana (F5) para poder finalizar la sesion. **OJO** Esto solo es necesario hacerlo si necesitamos una GPU de un tamaño de memoria especifico. Si nuestro proyecto es pequeño, cualquier GPU asignada por Colab servirá.

Nota: Este notebook está pre-configurado para trabajar con 25 Gb de memoria RAM, sin embargo esta configuración no es por defecto. La cantidad de memoria asignada por general es de 12 a 16 GB. Si crea un notebook nuevo en Colab y quiere tener 25 GB ver sección de Bonus al final de este tutorial. 


### Comandos Básicos
Debido a que cada notebook de Colab se ejecuta en una máquina Linux, es posible utilizar todos los comandos de Linux. Los comandos más básicos para movernos dentro de este tipo de ambiente son:

```
%pwd # Ver el directorio actual de trabajo
%ls # Listar los archivos del directorio actual
%cd # Cambiar de directorio
%mkdir # Crear un nuevo directorio
%rmdir # Eliminar un directorio vacio
```
Por ejemplo, si quiero ver en que carpeta me encuentro actualmente ejecuto una celda con el comando:


```
%pwd
```




    '/content'



Como vemos, me encuentro en la carpeta '/content', es decir, la carpeta raiz de Colab. Adicionalmente, podemos ejecutar varios comandos dentro de una celda agregando %%shell al inicio de esta. Por ejemplo:


```
%%shell
pwd
ls
mkdir "nuevo_directorio"
ls
rmdir "nuevo_directorio"
ls
```

    /content
    sample_data
    nuevo_directorio  sample_data
    sample_data





    



Note que si se agrega *%%shell* al inicio, se debe quitar el simbolo % al principio de cada comando. Así, en vez de escribir *%pwd*, escribimos *pwd* solamente. Note ademas, que el resultado de cada comando se muestra en una linea aparte. De esta manera, el primer comando ls, muestra el contenido de la carpeta /content, que en este caso es solo la carpeta 'sample_data/'. Luego creamos la carpeta 'nuevo_directorio' (observe que el comando mkdir no arroja ninguna salida), listamos el contenido de content/ para ver la nueva carpeta creada y por ultimo borramos la carpeta.

En general se puede utilizar cualquier comando linux, incluso instalar paquetes linux con el software apt, por ejemplo:


```
%%shell
sudo apt install nano
```


```
%%shell
sudo apt install nano
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    Suggested packages:
      spell
    The following NEW packages will be installed:
      nano
    0 upgraded, 1 newly installed, 0 to remove and 31 not upgraded.
    Need to get 231 kB of archives.
    After this operation, 778 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/main amd64 nano amd64 2.9.3-2 [231 kB]
    Fetched 231 kB in 2s (130 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package nano.
    (Reading database ... 144433 files and directories currently installed.)
    Preparing to unpack .../nano_2.9.3-2_amd64.deb ...
    Unpacking nano (2.9.3-2) ...
    Setting up nano (2.9.3-2) ...
    update-alternatives: using /bin/nano to provide /usr/bin/editor (editor) in auto mode
    update-alternatives: using /bin/nano to provide /usr/bin/pico (pico) in auto mode
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...





    



Puede buscar mas comandos en internet, una pagina inicial sería:

https://marcosmarti.org/comandos-basicos-de-linux/

## Setup

Lo primero que haremos será conectarnos con nuestra cuenta de Google Drive. **Importante conectarse a la cuenta de google drive en donde se tiene almacenado el proyecto o el código a ejecutar.** Adicionalmente, esta cuenta debe estar previamente sincronizada en nuestra computadora utilizando un cliente de Google Drive (Ver Introducción). Para conectarnos a google drive, ejecutamos la celda de abajo y seguimos los pasos. Por favor seleccione la cuenta previamente sincronizada.

### Mount Goolge Drive


```
# link to google drive

from google.colab import drive
#drive.mount('/content/gdrive/')
drive.mount("/content/gdrive/", force_remount=True)
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive/


Si damos click en el panel de la izquierda, al icono de carpeta, veremos que tenemos una nueva carpeta llamada 'gdrive'. De esta manera habremos sincronizado de manera correcta google drive con Colab.

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/ver_gdrive.png)



Ahora la idea es trabajar en nuestro código. Existen básicamente dos maneras:


1.   Añadir el codigo directamente en este notebook e ir ejecutando cada celda.
2.   Ejecutar un script de python directamente desde una celda.

En este tutorial se utilizará la segunda forma. Para esto utilicé como ejemplo el tutorial de tensorflow:

https://www.tensorflow.org/tensorboard/get_started

Seguidamente, creé un proyecto en mi computadora local llamado colab_tutorial. La carpeta de este proyecto se encuentra sincronizado con mi Google Drive, por lo tanto, todo cambio que se realice en mi computadora local se vera reflejado directamente en Drive y, de igual forma, en Colab.

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/screen_pycharm.png)

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/carpeta_google_drive.png)

Una vez tengamos listo el código en nuestra carpeta local, solo basta con ejecutar una celda de la siguiente forma


```
!python3 nombre_archivo.py --parametros (opcional)
```

De esta forma ejecutaremos nuestro código en Colab. Note que nuestro código tambien puede generar salidas, las cuales (si se almacenan dentro de la misma carpeta del proyecto) quedarán sincronizadas con nuestra computadora local.





Aqui un ejemplo de ejecutar la funcion main.py dentro de mi carpeta proyecto "colab_tutorial". Primero vamos hacia la carpeta con los comandos básicos explicados en la sección 2.


```
%cd /content/gdrive/My\ Drive/colab_tutorial/
```

    /content/gdrive/My Drive/colab_tutorial



```
%ls
```

    main.py



```
!python main.py
```

    2020-05-19 01:24:56.751869: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-19 01:24:58.852417: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-05-19 01:24:58.866042: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.866658: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
    coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
    2020-05-19 01:24:58.866696: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-19 01:24:58.868230: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-19 01:24:58.869913: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-19 01:24:58.870253: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-19 01:24:58.871814: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-19 01:24:58.872594: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-19 01:24:58.875551: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-19 01:24:58.875654: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.876241: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.876770: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-19 01:24:58.881933: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2200000000 Hz
    2020-05-19 01:24:58.882377: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x138f100 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-05-19 01:24:58.882409: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-05-19 01:24:58.960753: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.961607: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x138ef40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-05-19 01:24:58.961643: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P100-PCIE-16GB, Compute Capability 6.0
    2020-05-19 01:24:58.961839: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.962429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
    coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
    2020-05-19 01:24:58.962471: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-19 01:24:58.962526: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-19 01:24:58.962541: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-19 01:24:58.962554: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-19 01:24:58.962566: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-19 01:24:58.962581: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-19 01:24:58.962596: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-19 01:24:58.962655: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.963213: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:58.963746: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-19 01:24:58.963791: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-19 01:24:59.498013: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-05-19 01:24:59.498074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
    2020-05-19 01:24:59.498084: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
    2020-05-19 01:24:59.498276: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:59.498864: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-19 01:24:59.499412: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-05-19 01:24:59.499466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14973 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0)
    2020-05-19 01:24:59.547390: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
    2020-05-19 01:24:59.547453: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1363] Profiler found 1 GPUs
    2020-05-19 01:24:59.548374: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcupti.so.10.1
    2020-05-19 01:24:59.678606: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1479] CUPTI activity buffer flushed
    Epoch 1/10
    2020-05-19 01:25:00.110434: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-19 01:25:00.407164: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
       1/1875 [..............................] - ETA: 0s - loss: 2.5025 - accuracy: 0.0000e+002020-05-19 01:25:00.411242: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1479] CUPTI activity buffer flushed
    2020-05-19 01:25:00.411386: I tensorflow/core/profiler/internal/gpu/device_tracer.cc:216]  GpuTracer has collected 62 callback api events and 62 activity events.
    2020-05-19 01:25:00.424149: I tensorflow/core/profiler/rpc/client/save_profile.cc:168] Creating directory: logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00
    2020-05-19 01:25:00.430074: I tensorflow/core/profiler/rpc/client/save_profile.cc:174] Dumped gzipped tool data for trace.json.gz to logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00/998f21424347.trace.json.gz
    2020-05-19 01:25:00.430893: I tensorflow/core/profiler/utils/event_span.cc:288] Generation of step-events took 0.016 ms
    
    2020-05-19 01:25:00.449145: I tensorflow/python/profiler/internal/profiler_wrapper.cc:87] Creating directory: logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00Dumped tool data for overview_page.pb to logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00/998f21424347.overview_page.pb
    Dumped tool data for input_pipeline.pb to logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00/998f21424347.input_pipeline.pb
    Dumped tool data for tensorflow_stats.pb to logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00/998f21424347.tensorflow_stats.pb
    Dumped tool data for kernel_stats.pb to logs/fit/20200519-012459/train/plugins/profile/2020_05_19_01_25_00/998f21424347.kernel_stats.pb
    
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2168 - accuracy: 0.9365 - val_loss: 0.1001 - val_accuracy: 0.9694
    Epoch 2/10
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0948 - accuracy: 0.9710 - val_loss: 0.0865 - val_accuracy: 0.9725
    Epoch 3/10
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0670 - accuracy: 0.9786 - val_loss: 0.0708 - val_accuracy: 0.9778
    Epoch 4/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0525 - accuracy: 0.9832 - val_loss: 0.0623 - val_accuracy: 0.9810
    Epoch 5/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0433 - accuracy: 0.9855 - val_loss: 0.0696 - val_accuracy: 0.9806
    Epoch 6/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0357 - accuracy: 0.9885 - val_loss: 0.0685 - val_accuracy: 0.9816
    Epoch 7/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0299 - accuracy: 0.9901 - val_loss: 0.0625 - val_accuracy: 0.9822
    Epoch 8/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0276 - accuracy: 0.9908 - val_loss: 0.0655 - val_accuracy: 0.9821
    Epoch 9/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0237 - accuracy: 0.9925 - val_loss: 0.0789 - val_accuracy: 0.9806
    Epoch 10/10
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0230 - accuracy: 0.9923 - val_loss: 0.0750 - val_accuracy: 0.9827


En este ejemplo en particular se hizo uso de la herramienta [Tensorboard](https://www.tensorflow.org/tensorboard), el cual proporciona una interfaz de visualización para el entrenamiento de los modelos. Al ejecutar nuestro código, se guarda en la carpeta logs el resultado de entrenamiento para cada epoch. Para visualizar la herramienta Tensorboard tenemos basicamente dos formas:


1.   Visualizar Tensorboard directamente en Colab
2.   Visualizar Tensorboard en nuestra computadora local

La desventaja de visualizar en Colab es que debemos esperar a que termine el entrenamiento para visualizar Tensorboard. Por el contrario, debido a que tenemos nuestra carpeta sincronizada, podemos ejecutar Tensorboard en nuestra computadora y visualizar el entrenamiento de manera local en tiempo real mientras se ejecuta el entrenamiento en Colab.


Para visualizar el resultado directamente en este notebook ejecutamos las siguientes dos celdas. La primera solo se ejecuta una vez, pues es la encargada de cargar la extension.


```
# Load the TensorBoard notebook extension (Ejecutar una sola vez)
%load_ext tensorboard
```


```
%tensorboard --logdir logs/fit
```


    Reusing TensorBoard on port 6006 (pid 1921), started 0:04:55 ago. (Use '!kill 1921' to kill it.)



    <IPython.core.display.Javascript object>


Los detalles de implementación del código utilizado de ejemplo se pueden consultar directamente en la página del tutorial: https://www.tensorflow.org/tensorboard/get_started

## Conclusiones

### Conclusion
En conclusion, gracias a la sincronización que nos provee el cliente de google drive, podemos hacer todas las modificaciones necesarias de manera local en nuestras computadoras y unicamente utilizar colab para ejecturar el entrenamiento de la red neuronal aprovechando su GPU. Adicionalmente, el uso de la herramienta Tensorboard es fundamental para monitorear el entrenamiento.

### Bonus
**1)**
Cuando ejecutamos entrenamientos muy largos, es posible que Colab se desconecte inesperadamente debido a falta de interactividad. Para solucionar este problema podemos agregar un código Javascript en esta página para hacer "clicks" de manera automática en el boton Conect (Boton donde se muestra uso de RAM y Disco). El código en cuestión es el siguiente:



```
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```

Este código debe agregarse en la consola de Javascript de la siguiente manera:

![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/prevent_disconecting.gif)


**2)** Si se quiere mas memoria ram, se puede ejecutar el siguiente código en una celda:


```
a = []
while(1):
    a.append(‘1’)
```
Al ejecutarlo, este comando va a llenar la memoria ram disponible en Colab y forzará a este a ampliar la capacidad. Debemos esperar aproximadamente 1 minuto hasta que salga un letrero que pregunta si deseamos ampliar la memoria. Despues de aceptar, tendremos más memoria RAM en nuestro notebook. Mas información: https://towardsdatascience.com/upgrade-your-memory-on-google-colab-for-free-1b8b18e8791d


## Tarea
Realice las siguientes actividades y elabore un informe con sus análisis y conclusiones
1. Lea el tutorial descrito en el siguiente enlace:  https://www.tensorflow.org/tensorboard/get_started
2. Instale el IDE de python Pycharm y el cliente de Google drive como se describe en este tutorial.
3. Implemente el código mostrado en el tutorial https://www.tensorflow.org/tensorboard/get_started en un archivo de python (.py).
4. Sincronice el código con Google Drive utilizando el cliente de Google drive
5. Ejecute el código en Colab y visualice la salida en Tensorboard.
6. Utilice el editor Pycharm para modificar el archivo de python donde se implementó el código. Cambie el número de epocas (EPOCHS)
a 10 y vuelva a ejecutar el código.
7. Elabore un informe **corto** donde describa con sus palabras que es Google Colab y que es Tensorboard y para qué sirve estas dos
herramientas.


