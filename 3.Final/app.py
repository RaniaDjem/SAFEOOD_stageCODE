import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from typing import Tuple, Union
from deskew import determine_skew
import itertools
import re
from flask import Flask, request, jsonify
from flask import Flask, request, render_template, send_file
from io import BytesIO



# Retourne l'OCR
def ocr(img):
    # Convert image to the expected data type
    img = img.astype(np.uint8)

    processed_text = pytesseract.image_to_string(img)
    # processed_text = processed_text.replace('\n', ' ')
    # print(processed_text)
    return processed_text

# FONCTION QUI RETOURNE L'IMAGE EN GRAYSCALE
def conv_gray(img):
    if len(img.shape) == 3:
        if img.dtype == np.uint8:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray

def resize_image(image, scale_factor=1.2):
    # les dim de l'image d'origine
    height, width = image.shape[:2]

    # news dimensions en fonction du facteur d'échelle
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Redimensionne l'image en utilisant la nouvelle taille
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# ROTATION/DESKEWING
# https://pypi.org/project/deskew/
def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def deskewing(image):
    # niv de gris
    if len(image.shape) == 3:  # verif si img n'est pas en niveaux de gris
        gray = gray = conv_gray(image)
    else:
        gray = image
    # gray = conv_gray(image)
    angle = determine_skew(gray)
    rotated = rotate(image, angle, (255, 255, 255))
    return rotated


def adjust_brightness(image):
    # Appliquer une transformation racine carrée pour ajuster la luminosité
    adjusted_image = np.sqrt(image)
    adjusted_image = (255 * (adjusted_image / np.max(adjusted_image))).astype(np.uint8)
    return adjusted_image


def invert_colors(image):
    # Inverser les couleurs de l'image en utilisant le complément de 255 (blanc - couleur = couleur inversée)
    inverted_image = 255 - image
    return inverted_image


def apply_adaptive_binarization(image, block_size=11, constant=2):
    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:  # Vérifier si img est en couleur (3 canaux)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # img déjà en niveaux de gris

    # Apply une binarisation adaptative à img
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
    )
    return binary_image


def adjust_contrast(image, alpha=1.5, beta=0.5):
    # Ajuster le contraste de l'image après binarisation
    contrast_adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_adjusted_image


def improve_letter_readability(image, kernel_size=(3, 3), iterations=1):
    """
    Applique une transformation morphologique d'érosion pour améliorer la lisibilité des lettres d'une image.
    :param kernel_size: Taille du noyau pour l'érosion (par défaut : (3, 3)).
    :return: L'image avec la lisibilité améliorée.
    """
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) == 3:  # si img est en couleur (3 canaux)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # img en niveaux de gris

    # érosion
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(gray_image, kernel, iterations=iterations)
    return eroded_image


def preprocess_text(text):
    """
    Applique un prétraitement pour supprimer les lignes vides et les lignes non pertinentes.
    """
    # Supprime les lignes vides
    non_empty_lines = [line.strip() for line in text.split('\n') if line.strip() != '']

    # Supprime les lignes contenant une seule lettre ou des suites de lettres séparées par des espaces
    filtered_lines = []
    for line in non_empty_lines:
        if len(line) <= 1 or re.match(r'^[a-zA-Z]+( [a-zA-Z]+)*$', line):
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def count_lines(text):
    return len(text.split('\n'))


def count_lines_in_texts(text1, text2, text3):
    num_lines_text1 = count_lines(text1)
    num_lines_text2 = count_lines(text2)
    num_lines_text3 = count_lines(text3)

    return num_lines_text1, num_lines_text2, num_lines_text3


def find_text_with_most_lines(texts):
    lines_counts = [count_lines(text) for text in texts]
    max_lines = max(lines_counts)
    texts_with_max_lines = [text for text, count in zip(texts, lines_counts) if count == max_lines]

    if len(texts_with_max_lines) == 1:
        return texts_with_max_lines[0]
    else:
        max_chars = max(len(text) for text in texts_with_max_lines)
        return next(text for text in texts_with_max_lines if len(text) == max_chars)




def meilleur_text(fonctions_pretraitement, image_data):
    meilleur_resultat = 0
    meilleur_texte = ""

    image_array = np.array(image_data, dtype=np.uint8)

    # convert image data to a NumPy array
    #image_array = np.frombuffer(image_data, dtype=np.uint8)

    # OCR sans prétraitement
    ocr_result_sans_pretraitement = ocr(image_array)
    preprocessed_text_sans = preprocess_text(ocr_result_sans_pretraitement)
    num_lines_sans = count_lines(preprocessed_text_sans)
    meilleur_resultat = num_lines_sans
    meilleur_texte = preprocessed_text_sans

    # Appliquer les fonctions de prétraitement
    image_array = deskewing(adjust_contrast(conv_gray(image_array)))

    # Générer toutes les combinaisons possibles de fonctions de prétraitement
    for n in range(1, len(fonctions_pretraitement) + 1):
        combinaisons = list(itertools.combinations(fonctions_pretraitement, n))

        # Appliquer les fonctions de prétraitement dans chaque combinaison
        for combinaison in combinaisons:
            donnees_pretraitees = image_array.copy()
            for fonction in combinaison:
                donnees_pretraitees = fonction(donnees_pretraitees)

            # OCR avec Tesseract
            ocr_result = ocr(donnees_pretraitees)

            # Prétraitement du texte OCR
            preprocessed_text = preprocess_text(ocr_result)

            # Comptage des lignes
            num_lines = count_lines(preprocessed_text)

            # Mettre à jour le meilleur texte en fonction du nombre de lignes
            if num_lines > meilleur_resultat:
                meilleur_resultat = num_lines
                meilleur_texte = preprocessed_text

    return meilleur_texte


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Lire l'image téléchargée depuis la requête
            image = Image.open(BytesIO(uploaded_file.read()))
            bw_image = image.convert('L')  # Convertir en noir et blanc
            output_buffer = BytesIO()
            bw_image.save(output_buffer, format="JPEG")
            output_buffer.seek(0)

            # Envoyer l'image résultante en tant que réponse
            return send_file(output_buffer, mimetype='image/jpeg')
        else:
            return jsonify(success=False, message="Aucun fichier téléchargé.")
    except Exception as e:
        return jsonify(success=False, message=str(e))


@app.route('/convert_to_text', methods=['POST'])
def convert_image_to_text():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Lire l'image téléchargée avec PIL
            #image = Image.open(uploaded_file)
            image = np.array(Image.open(uploaded_file))

            #MODIF
            meilleur_resultat = 0
            meilleur_text = ""

            #OCR without any preprocessing
            ocr_result_sans_pretraitement = ocr(image)
            preprocessed_text_sans = preprocess_text(ocr_result_sans_pretraitement)
            num_lines_sans = count_lines(preprocessed_text_sans)
            
            meilleur_resultat = num_lines_sans
            meilleur_texte = preprocessed_text_sans


            image = deskewing(adjust_contrast(conv_gray(image)))

            fonctions_pretraitement = [
                resize_image,
                invert_colors,
                improve_letter_readability,
                adjust_brightness,
            ]

            # Générer toutes les combinaisons possibles de fonctions de prétraitement
            for n in range(1, len(fonctions_pretraitement) + 1):
                combinaisons = list(itertools.combinations(fonctions_pretraitement, n))

                # Appliquer les fonctions de prétraitement dans chaque combinaison
                for combinaison in combinaisons:
                    donnees_pretraitees = image.copy()
                    for fonction in combinaison:
                        donnees_pretraitees = fonction(donnees_pretraitees)
                        #print('OK')
                    # OCR avec Tesseract
                    ocr_result = ocr(donnees_pretraitees)

                    # Prétraitement du texte OCR
                    preprocessed_text = preprocess_text(ocr_result)

                    # Comptage des lignes
                    num_lines = count_lines(preprocessed_text)

                    # Mettre à jour le meilleur texte en fonction du nombre de lignes
                    if num_lines > meilleur_resultat:
                        meilleur_resultat = num_lines
                        meilleur_texte = preprocessed_text

            #ocr_text = ocr(np.array(image))

            return meilleur_texte

        else:
            return "Aucun fichier téléchargé."
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run()



