import numpy as np
import cv2
import face_recognition as fr
import subprocess

# Inicia a captura de vídeo
cap = cv2.VideoCapture("")#endereço ip da sua camera no formato https://...

# Carrega a imagem e gera a codificação do rosto
my_img = fr.load_image_file("#imagem teste")
my_face_encoding = fr.face_encodings(my_img)[0]  # Obtenha apenas o primeiro rosto detectado

# Armazena codificações e nomes conhecidos
know_face_encodings = [my_face_encoding]
know_face_names = [""] # Nome da pessoa da imagem teste

# Inicializa variáveis
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo.")
        break
    
    # Processa cada segundo frame para economizar recursos
    if process_this_frame:
        # Reduz o tamanho do frame para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Localiza e codifica rostos no frame atual
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame)

        # Verifica correspondências
        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(know_face_encodings, face_encoding)
            name = "Unknown"

            # Calcula a distância e encontra a melhor correspondência
            face_distances = fr.face_distance(know_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = know_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Exibe os resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Ajusta as coordenadas para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha um retângulo ao redor 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Adiciona o nome do rosto reconhecido
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostra o frame na janela
    cv2.imshow('Video', frame)

    # Sai do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
