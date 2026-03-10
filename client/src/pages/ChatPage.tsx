import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Box,
  Button,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  TextField,
  Typography,
} from "@mui/material";

import { createChat, getChat, listChats, sendMessage } from "../lib/api";
import type { Chat, Message } from "../types";
import { useAuthStore } from "../store/auth";

function ChatPage() {
  const { chatId } = useParams();
  const navigate = useNavigate();
  const logout = useAuthStore((state) => state.logout);
  const user = useAuthStore((state) => state.user);

  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeChatId = useMemo(() => (chatId ? Number(chatId) : null), [chatId]);

  useEffect(() => {
    const loadChats = async () => {
      setLoading(true);
      try {
        const data = await listChats();
        setChats(data);
        if (!activeChatId && data.length > 0) {
          navigate(`/chat/${data[0].id}`, { replace: true });
        }
      } catch {
        setError("Failed to load chats.");
      } finally {
        setLoading(false);
      }
    };

    void loadChats();
  }, [activeChatId, navigate]);

  useEffect(() => {
    const loadChatDetail = async () => {
      if (!activeChatId) {
        setMessages([]);
        return;
      }

      setLoading(true);
      try {
        const chat = await getChat(activeChatId);
        setMessages(chat.messages);
      } catch {
        setError("Failed to load chat history.");
      } finally {
        setLoading(false);
      }
    };

    void loadChatDetail();
  }, [activeChatId]);

  const handleNewChat = async () => {
    setError(null);
    try {
      const created = await createChat("New Chat");
      setChats((prev) => [created, ...prev]);
      navigate(`/chat/${created.id}`);
    } catch {
      setError("Unable to create chat.");
    }
  };

  const handleSend = async () => {
    if (!input.trim()) {
      return;
    }
    setSending(true);
    setError(null);

    try {
      let targetChatId = activeChatId;
      if (!targetChatId) {
        const created = await createChat("New Chat");
        setChats((prev) => [created, ...prev]);
        targetChatId = created.id;
        navigate(`/chat/${created.id}`);
      }

      const newMessages = await sendMessage(targetChatId, input.trim());
      setMessages((prev) => [...prev, ...newMessages]);
      setInput("");
    } catch {
      setError("Unable to send message.");
    } finally {
      setSending(false);
    }
  };

  return (
    <Box display="flex" height="100vh">
      <Box
        width={280}
        borderRight="1px solid"
        borderColor="divider"
        padding={2}
        display="flex"
        flexDirection="column"
        gap={2}
      >
        <Stack spacing={0.5}>
          <Typography variant="h6">MarketLogic AI</Typography>
          <Typography variant="body2" color="text.secondary">
            {user?.full_name ?? user?.username}
          </Typography>
        </Stack>
        <Button variant="contained" onClick={handleNewChat}>
          New Chat
        </Button>
        <Divider />
        <Box flex={1} overflow="auto">
          <List disablePadding>
            {chats.map((chat) => (
              <ListItemButton
                key={chat.id}
                selected={chat.id === activeChatId}
                onClick={() => navigate(`/chat/${chat.id}`)}
              >
                <ListItemText primary={chat.title} secondary={`#${chat.id}`} />
              </ListItemButton>
            ))}
            {chats.length === 0 && (
              <Typography color="text.secondary" padding={1}>
                No chats yet.
              </Typography>
            )}
          </List>
        </Box>
        <Button variant="text" color="inherit" onClick={logout}>
          Logout
        </Button>
      </Box>

      <Box flex={1} display="flex" flexDirection="column" padding={3} gap={2}>
        <Box flex={1} overflow="auto" display="flex" flexDirection="column" gap={2}>
          {loading && <Typography color="text.secondary">Loading...</Typography>}
          {error && <Typography color="error">{error}</Typography>}
          {messages.map((message) => (
            <Box
              key={message.id}
              alignSelf={message.role === "user" ? "flex-end" : "flex-start"}
              maxWidth="70%"
            >
              <Box
                padding={2}
                borderRadius={2}
                bgcolor={message.role === "user" ? "primary.main" : "grey.200"}
                color={message.role === "user" ? "primary.contrastText" : "text.primary"}
              >
                <Typography variant="body1">{message.content}</Typography>
              </Box>
            </Box>
          ))}
          {messages.length === 0 && !loading && (
            <Typography color="text.secondary">
              Start a new evaluation by sharing the film logline and key metrics.
            </Typography>
          )}
        </Box>
        <Stack direction="row" spacing={2}>
          <TextField
            fullWidth
            placeholder="Ask MarketLogic about a film..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void handleSend();
              }
            }}
          />
          <Button variant="contained" onClick={handleSend} disabled={sending}>
            {sending ? "Sending..." : "Send"}
          </Button>
        </Stack>
      </Box>
    </Box>
  );
}

export default ChatPage;
