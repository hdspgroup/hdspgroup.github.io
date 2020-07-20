---
# Course title, summary, and position.
linktitle: Deep Learning
summary: Laboratorios de Deep Learning correspondientes al periodo de prueba para ingresar al grupo HDSP de la Universidad Industrial de Santander.
weight: 1

# Page metadata.
title: HDSP - Laboratorio de Deep Learning
date: "2020-06-02T00:00:00Z"
lastmod: "2020-06-02T00:00:00Z"
draft: false  # Is this a draft? true/false
toc: true  # Show table of contents? true/false
type: docs  # Do not modify.

# Add menu entry to sidebar.
# - name: Declare this menu item as a parent with ID `name`.
# - weight: Position of link in menu.
menu:
  deep-learning-labs:
    name: Vista General
    weight: 1
---
## Introducción

El objetivo de este laboratorio es brindar al estudiante HDSP las herramientas base 
para el desarrollo de algoritmos de aprendizaje profundo (Deep Learning en inglés) que 
le permitan solucionar distintas tareas de visión por computadora como detección y 
clasificación de objetos, así como tambien resolver problemas inversos como la 
recuperación o reconstrucción de imágenes espectrales comprimidas. 
El laboratorio estará dividio en x secciones donde el estudiante deberá 
resolver y/o ejecutar los respectivos ejercicios y responder preguntas o 
analizar los resultados obtenidos al final de cada sección.

## Desarrollo de los Laboratorios
Todos los laboratorios de Deep Learning serán entregados al estudiante HDSP en 
formato notebook de Python a través de la plataforma de [Google Colaboratory](https://colab.research.google.com/).
La principal razón para realizar los laboratorios en este formato es que Google Colaboratory, 
o simplemente Colab, permite ejecutar y programar en Python directamente desde el navegador. De esta manera
se facilita la configuración de un entorno para Deep Learning y el accesso gratuito a GPUs para ejecutar los algoritmos
desarrollados.

Antes de iniciar cada laboratorio, el estudiante debe dar click en el boton 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() 
lo cual abrirá el notebook, correspondiente al laboratorio, directamente en Colab. Luego, para poder editar y ejecutar
el código, el estudiante debe copiar el notebook en su Google Drive. Para esto, debe tener abierto el notebook en
Colab y luego dar click en el boton
![texto alternativo](https://raw.githubusercontent.com/carlosh93/carlosh93.github.io/master/files/notebook_files/copy_drive.png)

Una vez completado el anterior procedimiento, el estudiante debe seguir la guía proporcionada donde se abordarán
temas desde un nivel básico hasta un nivel intermedio. Al final de cada guía el estudiante deberá responder preguntas 
y/o resolver ejercicios. Un vez completado los ejercicios, el estudiante deberá compartir el notebook a su tutor y 
preparar un documento donde se resuma lo aprendido y se analicen los resultados obtenidos.

<!--
This feature can be used for publishing content such as:

* **Online courses**
* **Project or software documentation**
* **Tutorials**

The `courses` folder may be renamed. For example, we can rename it to `docs` for software/project documentation or `tutorials` for creating an online course.

## Delete tutorials

**To remove these pages, delete the `courses` folder and see below to delete the associated menu link.**

## Update site menu

After renaming or deleting the `courses` folder, you may wish to update any `[[main]]` menu links to it by editing your menu configuration at `config/_default/menus.toml`.

For example, if you delete this folder, you can remove the following from your menu configuration:

```toml
[[main]]
  name = "Courses"
  url = "courses/"
  weight = 50
```

Or, if you are creating a software documentation site, you can rename the `courses` folder to `docs` and update the associated *Courses* menu configuration to:

```toml
[[main]]
  name = "Docs"
  url = "docs/"
  weight = 50
```

## Update the docs menu

If you use the *docs* layout, note that the name of the menu in the front matter should be in the form `[menu.X]` where `X` is the folder name. Hence, if you rename the `courses/example/` folder, you should also rename the menu definitions in the front matter of files within `courses/example/` from `[menu.example]` to `[menu.<NewFolderName>]`.
-->
