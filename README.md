Rodar projeto backend:
    - Swagger:
        cd backend
        uvicorn app.main:app --reload
    - Server WebSocket
        cd backend
        uvicorn live_biometry_server:live_face_app --reload --host 0.0.0.0 --port 8004 // porta 8004 para coincidir com o front


Rodar projeto frontend:
    cd frontend
    npm run dev

Rode o setup_db para montar o bd no docker