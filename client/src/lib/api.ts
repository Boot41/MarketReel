import axios from "axios";

import type { Chat, ChatDetail, Message, User } from "../types";

const api = axios.create({
  baseURL: "/",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const response = await api.post<LoginResponse>("/auth/login", { username, password });
  return response.data;
}

export async function listChats(): Promise<Chat[]> {
  const response = await api.get<Chat[]>("/api/v1/chats");
  return response.data;
}

export async function createChat(title: string): Promise<Chat> {
  const response = await api.post<Chat>("/api/v1/chats", { title });
  return response.data;
}

export async function getChat(chatId: number): Promise<ChatDetail> {
  const response = await api.get<ChatDetail>(`/api/v1/chats/${chatId}`);
  return response.data;
}

export async function sendMessage(chatId: number, content: string): Promise<Message[]> {
  const response = await api.post<Message[]>(`/api/v1/chats/${chatId}/messages`, { content });
  return response.data;
}

export default api;
