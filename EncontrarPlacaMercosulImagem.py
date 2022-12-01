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

        # area de localização u 720p
        #area = frame[500:, 300:800]

        # area de localização 480p
        #area = frame[300:, 220:800]
        #area = frame[300:400, 200:300];
        #area = frame[150:350, 180:320]; certo
        area = frame[350:450, 280:500];

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
    img_roi = cv2.imread("resource/clio3.jpeg")
    imagem = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))

    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("output/roi-ocr-gray.png", imagem_gray)

    #imagem_suavizada = cv2.blur(imagem_gray, (1, 1))
    #cv2.imwrite("output/roi-ocr-suavizada.png", imagem_suavizada)


    _, imagem_limiarizada = cv2.threshold(imagem_gray, 120, 255, cv2.THRESH_BINARY)

    cv2.imwrite("output/roi-ocr-limiarizada.png", imagem_limiarizada)

    # desfoque
    #imagem_desfoque = cv2.GaussianBlur(imagem_limiarizada, (5, 5), 0)

    #cv2.imwrite("output/roi-ocr-desfoque.png", imagem_desfoque)
    imagem_suavizada = cv2.blur(imagem_limiarizada, (4,4))
    cv2.imwrite("output/roi-ocr-suavizada.png", imagem_suavizada)

    contornos, hier = cv2.findContours(imagem_suavizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contornos, hier = cv2.findContours(imagem_desfoque,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imagem_contornos = img_roi.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 2)
    cv2.imwrite("output/roi-ocr-contornos.png", imagem_contornos)
    for c in contornos:
        # perimetro do contorno, verifica se o contorno é fechado
        perimetro = cv2.arcLength(c, True)

        if perimetro > 900 and perimetro<10000:
            #print(perimetro);
            # aproxima os contornos da forma correspondente
            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            # verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
            if len(approx) == 4:
                # Contorna a placa atraves dos contornos encontrados
                (x, y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(img_roi, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                # segmenta a placa da imagem
                roi = img_roi[y:y + alt, x:x + lar]
                cv2.imwrite("output/roi-retangulo.png", roi)
   
    img_roi = cv2.imread("output/roi-retangulo.png")
    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Limiarização
    _, img = cv2.threshold(img, 99, 255, cv2.THRESH_BINARY)

    # Suavização da Imagem
    #img = cv2.blur(img, (4,4))

    # Aplica reconhecimento OCR no ROI com o Tesseract
    cv2.imwrite("output/roi-ocr.png", img)
    return img


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

    preProcessamentoRoi()

    reconhecimentoOCR()
