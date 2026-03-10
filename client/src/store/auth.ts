import { create } from "zustand";

import type { User } from "../types";

const tokenKey = "token";
const userKey = "user";

const storedToken = localStorage.getItem(tokenKey);
const storedUser = localStorage.getItem(userKey);

let initialUser: User | null = null;
if (storedUser) {
  try {
    initialUser = JSON.parse(storedUser) as User;
  } catch {
    initialUser = null;
  }
}

interface AuthState {
  token: string | null;
  user: User | null;
  login: (token: string, user: User) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: storedToken,
  user: initialUser,
  login: (token, user) => {
    localStorage.setItem(tokenKey, token);
    localStorage.setItem(userKey, JSON.stringify(user));
    set({ token, user });
  },
  logout: () => {
    localStorage.removeItem(tokenKey);
    localStorage.removeItem(userKey);
    set({ token: null, user: null });
  },
}));
