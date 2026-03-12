import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Divider,
  LinearProgress,
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

interface ParsedAssistantPayload {
  response_type?: string;
  message?: string;
  [key: string]: unknown;
}

function parseAssistantPayload(content: string): ParsedAssistantPayload | null {
  try {
    const parsed = JSON.parse(content) as unknown;
    if (parsed && typeof parsed === "object") {
      return parsed as ParsedAssistantPayload;
    }
    return null;
  } catch {
    return null;
  }
}

interface ScoreCardData {
  // Orchestrator-formatted fields
  film?: string;
  intent?: string;
  mg_recommendation?: { low_usd?: number; mid_usd?: number; high_usd?: number; rationale?: string };
  release_strategy?: { recommended_window?: string; notes?: string };
  comparable_films?: Array<{ title: string; territory_gross_usd: number }>;
  confidence_warning?: string | null;
  data_sufficiency_score?: number;
  citations?: Array<{ claim?: string; source_document?: string; page_or_chunk?: string; retrieved_by?: string }>;
  // ValuationAgent direct fields
  movie_id?: string;
  mg_estimate?: { low?: number; mid?: number; high?: number; confidence?: number; currency_code?: string; currency_usd?: number };
  theatrical_revenue_projection?: { low?: number; mid?: number; high?: number; currency?: string };
  vod_revenue_projection?: { low?: number; mid?: number; high?: number };
  comparable_films_used?: Array<{ title: string; territory?: string; mg?: number; similarity_score?: number }>;
  adjustments_applied?: string[];
  uncertainty_factors?: string[];
  sufficiency_note?: string;
  // Common fields
  territory?: string;
  risk_flags?: Array<{ flag?: string; severity?: string }>;
  [key: string]: unknown;
}

function ScoreCardView({ data }: { data: ScoreCardData }) {
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [showCitations, setShowCitations] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const fmt = (n?: number) =>
    n != null ? `$${(n / 1_000_000).toFixed(2)}M` : "—";
  const severityColor = (s?: string): "error" | "warning" | "success" =>
    s === "high" ? "error" : s === "medium" ? "warning" : "success";

  // Normalize: support both orchestrator-formatted and ValuationAgent direct output
  const filmName = data.film ?? data.movie_id ?? "—";
  const territory = data.territory ?? "—";
  const intent = data.intent ?? "valuation";

  const mgLow = data.mg_recommendation?.low_usd ?? data.mg_estimate?.low;
  const mgMid = data.mg_recommendation?.mid_usd ?? data.mg_estimate?.mid;
  const mgHigh = data.mg_recommendation?.high_usd ?? data.mg_estimate?.high;
  const mgRationale = data.mg_recommendation?.rationale;
  const currencyCode = data.mg_estimate?.currency_code;

  const score = data.data_sufficiency_score ?? data.mg_estimate?.confidence ?? 0;
  const scoreColor = score >= 0.6 ? "success" : score >= 0.4 ? "warning" : "error";

  const theatricalProj = data.theatrical_revenue_projection;
  const vodProj = data.vod_revenue_projection;
  const comparables = data.comparable_films ?? data.comparable_films_used ?? [];
  const riskFlags = data.risk_flags ?? [];
  const uncertaintyFactors = data.uncertainty_factors ?? [];
  const adjustmentsApplied = data.adjustments_applied ?? [];
  const citations = data.citations ?? [];

  return (
    <Card variant="outlined" sx={{ minWidth: 320, maxWidth: 640 }}>
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">{filmName}</Typography>
            <Chip label={territory} size="small" variant="outlined" />
          </Stack>
          <Typography variant="caption" color="text.secondary">
            Intent: {intent}
          </Typography>

          <Box>
            <Typography variant="subtitle2" gutterBottom>MG Recommendation</Typography>
            <Stack direction="row" spacing={2}>
              <Box textAlign="center">
                <Typography variant="caption" color="text.secondary">Low</Typography>
                <Typography variant="body2" fontWeight="bold">{fmt(mgLow)}</Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="caption" color="text.secondary">Mid</Typography>
                <Typography variant="body2" fontWeight="bold">{fmt(mgMid)}</Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="caption" color="text.secondary">High</Typography>
                <Typography variant="body2" fontWeight="bold">{fmt(mgHigh)}</Typography>
              </Box>
            </Stack>
            {mgRationale && (
              <Typography variant="body2" mt={1} color="text.secondary">{mgRationale}</Typography>
            )}
            {currencyCode && currencyCode !== "USD" && (
              <Typography variant="caption" color="text.secondary">
                Currency: {currencyCode}
                {data.mg_estimate?.currency_usd != null && ` (USD rate: ${data.mg_estimate.currency_usd})`}
              </Typography>
            )}
          </Box>

          {(theatricalProj ?? vodProj) && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>Revenue Projections</Typography>
              {theatricalProj && (
                <Stack direction="row" spacing={2} mb={0.5}>
                  <Typography variant="caption" color="text.secondary" minWidth={70}>Theatrical</Typography>
                  <Typography variant="caption">{fmt(theatricalProj.low)} – {fmt(theatricalProj.high)}</Typography>
                  {theatricalProj.currency && theatricalProj.currency !== "USD" && (
                    <Typography variant="caption" color="text.secondary">({theatricalProj.currency})</Typography>
                  )}
                </Stack>
              )}
              {vodProj && (
                <Stack direction="row" spacing={2}>
                  <Typography variant="caption" color="text.secondary" minWidth={70}>VOD</Typography>
                  <Typography variant="caption">{fmt(vodProj.low)} – {fmt(vodProj.high)}</Typography>
                </Stack>
              )}
            </Box>
          )}

          <Box>
            <Stack direction="row" justifyContent="space-between" mb={0.5}>
              <Typography variant="caption">Data Sufficiency</Typography>
              <Typography variant="caption">{(score * 100).toFixed(0)}%</Typography>
            </Stack>
            <LinearProgress variant="determinate" value={score * 100} color={scoreColor} />
            {data.sufficiency_note && (
              <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
                {data.sufficiency_note}
              </Typography>
            )}
          </Box>

          {data.confidence_warning && (
            <Alert severity="warning" sx={{ py: 0 }}>{data.confidence_warning}</Alert>
          )}

          {data.release_strategy && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>Release Strategy</Typography>
              {data.release_strategy.recommended_window && (
                <Chip label={data.release_strategy.recommended_window} size="small" sx={{ mr: 1 }} />
              )}
              {data.release_strategy.notes && (
                <Typography variant="body2" mt={0.5} color="text.secondary">
                  {data.release_strategy.notes}
                </Typography>
              )}
            </Box>
          )}

          {comparables.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>Comparable Films</Typography>
              <Stack spacing={0.5}>
                {comparables.map((c, i) => (
                  <Stack key={i} direction="row" justifyContent="space-between">
                    <Typography variant="body2">{c.title}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {"territory_gross_usd" in c
                        ? fmt((c as { territory_gross_usd: number }).territory_gross_usd)
                        : "mg" in c && (c as { mg?: number }).mg != null
                        ? fmt((c as { mg: number }).mg)
                        : "—"}
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Box>
          )}

          {riskFlags.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>Risk Flags</Typography>
              <Stack direction="row" flexWrap="wrap" gap={0.5}>
                {riskFlags.map((r, i) => (
                  <Chip
                    key={i}
                    label={r.flag ?? "—"}
                    size="small"
                    color={severityColor(r.severity)}
                    variant="outlined"
                  />
                ))}
              </Stack>
            </Box>
          )}

          {(uncertaintyFactors.length > 0 || adjustmentsApplied.length > 0) && (
            <Box>
              <Button
                size="small"
                variant="text"
                onClick={() => setShowUncertainty((v) => !v)}
                sx={{ px: 0, textTransform: "none" }}
              >
                {showUncertainty ? "Hide" : "Show"} uncertainty & adjustments
              </Button>
              {showUncertainty && (
                <Stack spacing={1} mt={1}>
                  {uncertaintyFactors.length > 0 && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">Uncertainty factors</Typography>
                      {uncertaintyFactors.map((u, i) => (
                        <Typography key={i} variant="body2">• {u}</Typography>
                      ))}
                    </Box>
                  )}
                  {adjustmentsApplied.length > 0 && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">Adjustments applied</Typography>
                      {adjustmentsApplied.map((a, i) => (
                        <Typography key={i} variant="body2">• {a}</Typography>
                      ))}
                    </Box>
                  )}
                </Stack>
              )}
            </Box>
          )}

          {citations.length > 0 && (
            <Box>
              <Button
                size="small"
                variant="text"
                onClick={() => setShowCitations((v) => !v)}
                sx={{ px: 0, textTransform: "none" }}
              >
                {showCitations ? "Hide" : "Show"} citations ({citations.length})
              </Button>
              {showCitations && (
                <Stack spacing={0.5} mt={1}>
                  {citations.map((c, i) => (
                    <Box key={i}>
                      <Typography variant="caption" display="block">{c.claim}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {c.source_document}{c.page_or_chunk ? ` · ${c.page_or_chunk}` : ""}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              )}
            </Box>
          )}

          <Box>
            <Button
              size="small"
              variant="text"
              onClick={() => setShowRaw((v) => !v)}
              sx={{ px: 0, textTransform: "none" }}
            >
              {showRaw ? "Hide" : "Show"} raw data
            </Button>
            {showRaw && (
              <Box
                component="pre"
                sx={{ fontSize: 11, overflowX: "auto", mt: 1, p: 1, bgcolor: "grey.900", borderRadius: 1 }}
              >
                {JSON.stringify(data, null, 2)}
              </Box>
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

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
                bgcolor={message.role === "user" ? "primary.main" : "grey.900"}
                color={message.role === "user" ? "primary.contrastText" : "grey.100"}
              >
                {message.role === "assistant" ? (
                  (() => {
                    const payload = parseAssistantPayload(message.content);
                    if (!payload) {
                      return <Typography variant="body1">{message.content}</Typography>;
                    }
                    if (
                      typeof (payload.film ?? payload.movie_id) === "string" &&
                      (payload.mg_recommendation != null || payload.mg_estimate != null || payload.risk_flags != null || payload.release_strategy != null)
                    ) {
                      return <ScoreCardView data={payload as ScoreCardData} />;
                    }
                    return <Typography variant="body1">{String(payload.message ?? message.content)}</Typography>;
                  })()
                ) : (
                  <Typography variant="body1">{message.content}</Typography>
                )}
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
