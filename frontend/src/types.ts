export type User = { id: string; email: string };
export type Citation = { chunk_id: string; document_id: string; document_name: string; page: number | null; excerpt: string };
export type Message = {
  id: string; conversation_id: string; parent_message_id: string | null;
  role: "user" | "assistant"; content: string; citations: Citation[]; created_at: string;
};
export type Conversation = { id: string; title: string; active_leaf_id: string | null; created_at: string; updated_at: string };
export type ConversationDetail = Conversation & { messages: Message[]; document_ids: string[] };
export type Document = { id: string; name: string; mime_type: string; size_bytes: number; status: string; error: string; created_at: string };
