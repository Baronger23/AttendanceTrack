import os
from django.core.files.storage import Storage
from django.utils.deconstruct import deconstructible


@deconstructible
class SupabaseStorage(Storage):
    """Custom storage backend for Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = (
            os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            or os.getenv('SUPABASE_SERVICE_KEY')
            or os.getenv('SUPABASE_KEY')
        )
        self.bucket_name = os.getenv('SUPABASE_BUCKET_NAME', 'attendance-storage')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Initialize Supabase client lazily
        self._client = None
    
    @property
    def client(self):
        """Lazy load Supabase client"""
        if self._client is None:
            from supabase import create_client
            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client
    
    def _save(self, name, content):
        """Upload file to Supabase Storage"""
        try:
            # Read file content
            content.seek(0)  # Reset file pointer to beginning
            file_content = content.read()
            
            # Get content type
            content_type = getattr(content, 'content_type', 'application/octet-stream')
            
            # Upload to Supabase Storage (use upsert to overwrite if exists)
            response = self.client.storage.from_(self.bucket_name).upload(
                path=name,
                file=file_content,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            return name
        except Exception as e:
            # If file exists, try to update it
            try:
                self.client.storage.from_(self.bucket_name).update(
                    path=name,
                    file=file_content,
                    file_options={"content-type": content_type}
                )
                return name
            except:
                raise IOError(f"Error uploading to Supabase: {str(e)}")
    
    def url(self, name):
        """Get public URL for the file"""
        try:
            return self.client.storage.from_(self.bucket_name).get_public_url(name)
        except Exception:
            return f"{self.supabase_url}/storage/v1/object/public/{self.bucket_name}/{name}"
    
    def delete(self, name):
        """Delete file from Supabase Storage"""
        try:
            self.client.storage.from_(self.bucket_name).remove([name])
        except Exception:
            pass
    
    def exists(self, name):
        """Check if file exists"""
        try:
            files = self.client.storage.from_(self.bucket_name).list(path=name)
            return len(files) > 0
        except Exception:
            return False
    
    def size(self, name):
        """Return the total size, in bytes, of the file"""
        try:
            files = self.client.storage.from_(self.bucket_name).list(path=name)
            if files and len(files) > 0:
                return files[0].get('metadata', {}).get('size', 0)
        except Exception:
            pass
        return 0
