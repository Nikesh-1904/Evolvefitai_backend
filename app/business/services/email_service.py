# app/business/services/email_service.py
"""Email notification service using Gmail SMTP"""

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending email notifications via Gmail SMTP"""

    def __init__(self):
        """Initialize email service with SMTP settings"""
        self.smtp_host = getattr(settings, 'SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.smtp_user = getattr(settings, 'SMTP_USER', '')
        self.smtp_password = getattr(settings, 'SMTP_PASSWORD', '')
        self.sender_email = getattr(settings, 'SENDER_EMAIL', self.smtp_user)
        self.sender_name = getattr(settings, 'SENDER_NAME', 'EvolveFit AI')

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None
    ) -> bool:
        """
        Send an email using Gmail SMTP
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body_html: HTML email body
            body_text: Plain text fallback (optional)
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = f"{self.sender_name} <{self.sender_email}>"
            message['To'] = to_email
            
            # Add plain text version if provided
            if body_text:
                part1 = MIMEText(body_text, 'plain')
                message.attach(part1)
            
            # Add HTML version
            part2 = MIMEText(body_html, 'html')
            message.attach(part2)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_user,
                password=self.smtp_password,
            )
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    async def send_fee_reminder(
        self,
        to_email: str,
        user_name: str,
        amount: float,
        due_date: datetime,
        gym_name: str
    ) -> bool:
        """
        Send fee payment reminder email
        
        Args:
            to_email: User's email
            user_name: User's full name
            amount: Fee amount
            due_date: Payment due date
            gym_name: Gym name
        
        Returns:
            True if sent successfully
        """
        subject = f"Fee Payment Reminder - {gym_name}"
        
        # Calculate days remaining
        days_remaining = (due_date.date() - datetime.utcnow().date()).days
        
        body_html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #4F46E5;">Fee Payment Reminder</h2>
                    
                    <p>Dear {user_name},</p>
                    
                    <p>This is a friendly reminder that your gym membership fee is due soon.</p>
                    
                    <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <p style="margin: 5px 0;"><strong>Gym:</strong> {gym_name}</p>
                        <p style="margin: 5px 0;"><strong>Amount Due:</strong> ₹{amount:.2f}</p>
                        <p style="margin: 5px 0;"><strong>Due Date:</strong> {due_date.strftime('%d %B %Y')}</p>
                        <p style="margin: 5px 0;"><strong>Days Remaining:</strong> {days_remaining} days</p>
                    </div>
                    
                    <p>Please ensure timely payment to continue enjoying uninterrupted access to gym facilities.</p>
                    
                    <p>If you have already made the payment, please ignore this reminder.</p>
                    
                    <p style="margin-top: 30px;">
                        Best regards,<br>
                        <strong>{gym_name} Team</strong><br>
                        <em>Powered by EvolveFit AI</em>
                    </p>
                    
                    <hr style="border: none; border-top: 1px solid #E5E7EB; margin: 30px 0;">
                    <p style="font-size: 12px; color: #6B7280;">
                        This is an automated message. Please do not reply to this email.
                    </p>
                </div>
            </body>
        </html>
        """
        
        body_text = f"""
        Fee Payment Reminder
        
        Dear {user_name},
        
        This is a friendly reminder that your gym membership fee is due soon.
        
        Gym: {gym_name}
        Amount Due: ₹{amount:.2f}
        Due Date: {due_date.strftime('%d %B %Y')}
        Days Remaining: {days_remaining} days
        
        Please ensure timely payment to continue enjoying uninterrupted access to gym facilities.
        
        If you have already made the payment, please ignore this reminder.
        
        Best regards,
        {gym_name} Team
        Powered by EvolveFit AI
        """
        
        return await self.send_email(to_email, subject, body_html, body_text)

    async def send_fee_overdue_notice(
        self,
        to_email: str,
        user_name: str,
        amount: float,
        due_date: datetime,
        gym_name: str
    ) -> bool:
        """Send fee overdue notice"""
        subject = f"⚠️ Overdue Fee Payment - {gym_name}"
        
        days_overdue = (datetime.utcnow().date() - due_date.date()).days
        
        body_html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #DC2626;">⚠️ Overdue Fee Payment</h2>
                    
                    <p>Dear {user_name},</p>
                    
                    <p>Your gym membership fee payment is now <strong>overdue</strong>.</p>
                    
                    <div style="background-color: #FEE2E2; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #DC2626;">
                        <p style="margin: 5px 0;"><strong>Gym:</strong> {gym_name}</p>
                        <p style="margin: 5px 0;"><strong>Amount Due:</strong> ₹{amount:.2f}</p>
                        <p style="margin: 5px 0;"><strong>Due Date:</strong> {due_date.strftime('%d %B %Y')}</p>
                        <p style="margin: 5px 0;"><strong>Days Overdue:</strong> {days_overdue} days</p>
                    </div>
                    
                    <p>Please make the payment at the earliest to avoid suspension of your membership.</p>
                    
                    <p>For any queries, please contact the gym management.</p>
                    
                    <p style="margin-top: 30px;">
                        Best regards,<br>
                        <strong>{gym_name} Team</strong><br>
                        <em>Powered by EvolveFit AI</em>
                    </p>
                </div>
            </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, body_html)

    async def send_custom_notification(
        self,
        to_emails: List[str],
        subject: str,
        message: str,
        gym_name: str
    ) -> int:
        """
        Send custom notification to multiple users
        
        Args:
            to_emails: List of recipient emails
            subject: Email subject
            message: Email message body
            gym_name: Gym name
        
        Returns:
            Number of emails sent successfully
        """
        body_html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #4F46E5;">{subject}</h2>
                    
                    <div style="margin: 20px 0;">
                        {message}
                    </div>
                    
                    <p style="margin-top: 30px;">
                        Best regards,<br>
                        <strong>{gym_name} Team</strong><br>
                        <em>Powered by EvolveFit AI</em>
                    </p>
                </div>
            </body>
        </html>
        """
        
        success_count = 0
        for email in to_emails:
            if await self.send_email(email, subject, body_html):
                success_count += 1
        
        return success_count


# Singleton instance
email_service = EmailService()
