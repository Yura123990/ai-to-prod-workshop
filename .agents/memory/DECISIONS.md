# Architectural Decisions — BUSFLOW

1. **Modular Monolith Structure**:
   - `backend/app/` contains `core/`, `models/`, `schemas/`, `repositories/`, `services/`, `api/`, `bot/`, `workers/`.
   - Strict separation: HTTP routes delegate to services; services handle business rules and transactions and call repositories; repositories execute SQLAlchemy queries.
2. **Database & Concurrency (Overbooking Prevention)**:
   - Production uses PostgreSQL (`asyncpg` / `psycopg`). Tests support isolated PostgreSQL (when `TEST_DATABASE_URL` is provided) and an async SQLite test harness with serialized locking (`with_for_update()`) so unit/integration/concurrency tests run deterministically in any environment.
   - Booking creation uses `SELECT ... FOR UPDATE` on the `Trip` row inside an explicit database transaction, computes confirmed seats (`SUM(seat_count) WHERE status = 'CONFIRMED'`), checks `reserved + requested <= trip.capacity`, checks idempotency key (`idempotency_key` unique per user/booking), inserts `Booking` + `OutboxNotification`, and commits atomically.
3. **Telegram Authentication & Security**:
   - Backend validates `initData` via HMAC-SHA256 using `HMAC_SHA256("WebAppData", bot_token)` as secret key over sorted `key=value` pairs (excluding `hash`), verifies `auth_date` expiration (`MAX_AUTH_AGE_SECONDS`), upserts `User`, and issues a signed Bearer token (or accepts `X-Telegram-Init-Data` directly) for subsequent API calls.
4. **Reliable Notifications & Outbox Pattern**:
   - All notifications (booking confirmed, booking cancelled, trip cancelled/rescheduled, waitlist seat available) are written to `notification_outbox` within the business transaction.
   - Background worker processes pending outbox entries asynchronously via Telegram Bot HTTP API with retry backoff; failures never roll back committed bookings.
5. **Unresolved Product Decisions (Explicit Defaults for MVP)**:
   - **Cancellation Deadline**: Configurable via `CANCELLATION_DEADLINE_MINUTES` (default `0`, meaning passengers can cancel any time before departure).
   - **Waitlist Promotion**: Notification-based workflow (`NOTIFY_ONLY`). When a booking is cancelled on a trip with `WAITING` entries that fit the newly available seats, matching waitlist entries are notified via Telegram outbox (`NOTIFIED` status) to book seats; seats are not auto-reserved.
   - **Default Timezone**: Configurable via `APP_TIMEZONE` (default `Europe/Kyiv`), supporting DST transitions via `zoneinfo.ZoneInfo`.
   - **Initial Admin Bootstrap**: Configurable via `ADMIN_TELEGRAM_IDS` env var so specified Telegram user IDs automatically receive `ADMIN` role on first authentication.
