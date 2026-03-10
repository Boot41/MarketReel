export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
}

export interface Chat {
  id: number;
  title: string;
  adk_session_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface ChatDetail extends Chat {
  messages: Message[];
}
