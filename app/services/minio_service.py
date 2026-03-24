import io
from minio import Minio
from minio.error import S3Error
from config import Config


class MinioService:
    """Handles file storage in MinIO object store."""

    def __init__(self):
        self.client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=Config.MINIO_SECURE,
        )
        self.bucket = Config.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            print(f"[MinIO] Warning: could not ensure bucket: {e}")

    def upload_file(self, doc_id: str, filename: str, file_data: bytes) -> str:
        """Upload a file to MinIO. Returns the object path."""
        object_name = f"{doc_id}/{filename}"
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(file_data),
            length=len(file_data),
            content_type="application/octet-stream",
        )
        return object_name

    def download_file(self, doc_id: str, filename: str) -> bytes:
        """Download a file from MinIO. Returns raw bytes."""
        object_name = f"{doc_id}/{filename}"
        response = self.client.get_object(self.bucket, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def delete_document(self, doc_id: str):
        """Delete all objects for a document ID."""
        objects = self.client.list_objects(self.bucket, prefix=f"{doc_id}/", recursive=True)
        for obj in objects:
            self.client.remove_object(self.bucket, obj.object_name)

    def is_healthy(self) -> bool:
        """Check if MinIO is reachable."""
        try:
            self.client.bucket_exists(self.bucket)
            return True
        except Exception:
            return False
