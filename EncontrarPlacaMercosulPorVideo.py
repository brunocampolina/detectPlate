import pytesseract
import matplotlib.pyplot as plt
import cv2


def desenhaContornos(contornos, imagem):
    for c in contornos:
        # perimetro do contorno, verifica se o contorno é fechado
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            # aproxima os contornos da forma correspondente
            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            # verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
            if len(approx) == 4:
                # Contorna a placa atraves dos contornos encontrados
                (x, y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                # segmenta a placa da imagem
                roi = imagem[y:y + alt, x:x + lar]
                cv2.imwrite("output/roi.png", roi)


def buscaRetanguloPlaca(source):
    # Captura ou Video
    video = cv2.VideoCapture(source)

    while video.isOpened():

        ret, frame = video.read()

        if (ret == False):
            break

        area = frame[330:400, 280:500];

        # escala de cinza
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # limiarização
        ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

        # desfoque
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        # lista os contornos
        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # limite horizontal
        cv2.line(frame, (0, 500), (1280, 500), (0, 0, 255), 1)
        # limite vertical 1
        cv2.line(frame, (300, 0), (300, 720), (0, 0, 255), 1)
        # limite vertical 2
        cv2.line(frame, (800, 0), (800, 720), (0, 0, 255), 1)

        cv2.imshow('FRAME', frame)

        desenhaContornos(contornos, area)

        cv2.imshow('RES', area)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    video.release()
    preProcessamentoRoi()
    cv2.destroyAllWindows()


def preProcessamentoRoi():
    img_roi = cv2.imread("output/roi.png")
    imagem = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))

    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("output/roi-ocr-gray.png", imagem_gray)

    #imagem_suavizada = cv2.blur(imagem_gray, (1, 1))
    #cv2.imwrite("output/roi-ocr-suavizada.png", imagem_suavizada)


    _, imagem_limiarizada = cv2.threshold(imagem_gray, 115, 255, cv2.THRESH_BINARY)

    cv2.imwrite("output/roi-ocr-limiarizada.png", imagem_limiarizada)

    imagem_suavizada = cv2.blur(imagem_limiarizada, (4,4))
    cv2.imwrite("output/roi-ocr-suavizada.png", imagem_suavizada)

    # Aplica reconhecimento OCR no ROI com o Tesseract
    cv2.imwrite("output/roi-ocr.png", imagem_suavizada)
    return imagem_suavizada


def reconhecimentoOCR():
    img_roi_ocr = cv2.imread("output/roi-ocr.png")
    if img_roi_ocr is None:
        return

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)

    print("Placa identificada: ")
    print(saida)
    return saida


if __name__ == "__main__":
    source = "resource/clio3 (1).mp4"
    
    buscaRetanguloPlaca(source)

    preProcessamentoRoi()

    reconhecimentoOCR()
